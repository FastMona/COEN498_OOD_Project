function result = MD3_chex(mode, net, varargin)
% MD3_chex  One-class latent-space Mahalanobis OOD post-filter (Stage 3).
%
%   Implements the same three algorithms as MD_Stage3_Postfilter but
%   adapted for the CheXpert pipeline:
%     - One-class training: global mean + covariance (no per-class split).
%     - Data loaded from imageDatastore (JPEG folder) instead of IDX files.
%     - Both CNN_chex (conv layers) and MLP_chex (flattenLayer + FC) use
%       imageInputLayer, so GAP collapses either H×W×C×N feature maps or
%       1×1×D×N FC outputs to N×D identically.
%
%   'LHL'    Algorithm 1 — Last hidden layer, single global MD.
%   'FUSION' Algorithm 2 — Multi-layer z-score fusion, single global MD.
%   'MBM'    Algorithm 3 — Multi-branch, one detector per branch;
%                          sample is OOD if ANY branch fires.
%
%   Threshold = 97.5th percentile of training MD scores per algorithm.
%
%   ── TRAIN ──────────────────────────────────────────────────────────────
%   model = MD3_chex('train', net, networkID, chexRoot)
%   model = MD3_chex('train', net, networkID, chexRoot, algorithm)
%   model = MD3_chex('train', net, networkID, chexRoot, algorithm, layerConfig)
%   model = MD3_chex('train', net, networkID, chexRoot, algorithm, layerConfig, forceRetrain)
%
%   ── TEST ───────────────────────────────────────────────────────────────
%   result = MD3_chex('test', net, testImds, model)
%
%   testImds : imageDatastore  or  cell array of .jpg / .jpeg file paths
%
%   ── layerConfig ────────────────────────────────────────────────────────
%   'LHL'    : char,     e.g. 'relu5'
%   'FUSION' : cellstr,  e.g. {'relu1','relu2','relu3','relu4','relu5'}
%   'MBM'    : cell of cellstr, e.g. {{'relu1'},{'relu2'},{'relu3'},{'relu4','relu5'}}
%   Omit layerConfig to auto-detect from the network's ReLU layers.
%
%   ── result struct fields ───────────────────────────────────────────────
%   .Algorithm  .IsOOD  .Accepted  .AcceptedCount  .RejectedCount
%   .Scores  .Threshold   (+ .BranchIsOOD / .BranchModels for MBM)

	switch lower(mode)
		case 'train'
			result = doTrain(net, varargin{:});
		case 'test'
			result = doTest(net, varargin{:});
		otherwise
			error('MD3_chex:badMode', 'mode must be ''train'' or ''test''.');
	end
end

% =========================================================================
% TRAIN DISPATCH
% =========================================================================

function model = doTrain(net, networkID, chexRoot, algorithm, layerConfig, forceRetrain)
	if nargin < 2 || isempty(networkID)
		error('MD3_chex:noID', 'networkID is required.');
	end
	if nargin < 3 || isempty(chexRoot)
		error('MD3_chex:noRoot', 'chexRoot is required.');
	end
	if nargin < 4 || isempty(algorithm),    algorithm    = 'LHL';   end
	if nargin < 6 || isempty(forceRetrain), forceRetrain = false;   end

	algorithm = upper(algorithm);

	cacheFile = getCacheFile(networkID, algorithm);
	if ~forceRetrain && isfile(cacheFile)
		data = load(cacheFile, 'model');
		if isfield(data, 'model') && ...
				strcmp(data.model.NetworkID, networkID) && ...
				strcmp(data.model.Algorithm, algorithm)
			model = data.model;
			fprintf('  Loaded Stage3 cache  [%s | %s]\n', algorithm, networkID);
			return;
		end
	end

	if nargin < 5 || isempty(layerConfig)
		layerConfig = autoLayerConfig(net, algorithm);
	end

	imds = imageDatastore(chexRoot, ...
		'FileExtensions', {'.jpg', '.jpeg'}, ...
		'ReadFcn', @readAndPreprocess);
	if numel(imds.Files) == 0
		error('MD3_chex:noImages', 'No .jpg files found in chexRoot: %s', chexRoot);
	end
	fprintf('  Building Stage3 manifold  [%s | %s] on %d images...\n', ...
		algorithm, networkID, numel(imds.Files));

	switch algorithm
		case 'LHL'
			model = trainLHL(net, imds, layerConfig, networkID);
		case 'FUSION'
			model = trainFusion(net, imds, layerConfig, networkID);
		case 'MBM'
			model = trainMBM(net, imds, layerConfig, networkID);
		otherwise
			error('MD3_chex:badAlgo', 'algorithm must be LHL, FUSION, or MBM.');
	end

	save(cacheFile, 'model', '-v7.3');
	fprintf('  Saved Stage3 cache to: %s\n', cacheFile);
end

% =========================================================================
% TEST DISPATCH
% =========================================================================

function results = doTest(net, testInput, model)
	testImds = toImds(testInput);
	switch model.Algorithm
		case 'LHL'
			results = testLHL(net, testImds, model);
		case 'FUSION'
			results = testFusion(net, testImds, model);
		case 'MBM'
			results = testMBM(net, testImds, model);
		otherwise
			error('MD3_chex:badAlgoInModel', 'Unknown algorithm in model: %s', model.Algorithm);
	end
end

% =========================================================================
% ALGORITHM 1 — LHL
% =========================================================================

function model = trainLHL(net, imds, layerName, networkID)
	fprintf('  LHL: extracting layer ''%s''\n', layerName);
	F = feats(net, imds, layerName);
	[mu, invSigma, D] = buildGlobalStats(F, 1e-3);
	scores    = mdScores1(F, mu, invSigma);
	threshold = prctile(scores, 97.5);
	model = struct('NetworkID', networkID,  'Algorithm',   'LHL', ...
	               'LayerName', layerName,  'FeatureDim',  D, ...
	               'Mu',        mu,         'InvSigma',    invSigma, ...
	               'Threshold', threshold,  'TrainScores', scores);
end

function results = testLHL(net, testImds, model)
	F      = feats(net, testImds, model.LayerName);
	checkDim(F, model.FeatureDim, model.LayerName);
	scores = mdScores1(F, model.Mu, model.InvSigma);
	results = packResults('LHL', scores, scores > model.Threshold, model.Threshold);
end

% =========================================================================
% ALGORITHM 2 — FUSION
% =========================================================================

function model = trainFusion(net, imds, layerNames, networkID)
	fprintf('  Fusion: layers  {%s}\n', strjoin(layerNames, ', '));
	[F, fusionStats] = fusedFeats(net, imds, layerNames, []);
	[mu, invSigma, D] = buildGlobalStats(F, 1e-2);
	scores    = mdScores1(F, mu, invSigma);
	threshold = prctile(scores, 97.5);
	model = struct('NetworkID',   networkID,    'Algorithm',   'FUSION', ...
	               'LayerNames',  {layerNames}, 'FusionStats', fusionStats, ...
	               'FeatureDim',  D,            'Mu',          mu, ...
	               'InvSigma',    invSigma,     'Threshold',   threshold, ...
	               'TrainScores', scores);
end

function results = testFusion(net, testImds, model)
	F      = fusedFeats(net, testImds, model.LayerNames, model.FusionStats);
	checkDim(F, model.FeatureDim, 'fused');
	scores = mdScores1(F, model.Mu, model.InvSigma);
	results = packResults('FUSION', scores, scores > model.Threshold, model.Threshold);
end

% =========================================================================
% ALGORITHM 3 — MBM
% =========================================================================

function model = trainMBM(net, imds, branches, networkID)
	numBranches  = numel(branches);
	branchModels = cell(numBranches, 1);

	for b = 1:numBranches
		layerNames = branches{b};
		numL       = numel(layerNames);
		fprintf('  MBM branch %d/%d: {%s}\n', b, numBranches, strjoin(layerNames, ', '));

		layerStats = cell(numL, 1);
		scoreList  = cell(numL, 1);

		for k = 1:numL
			F = feats(net, imds, layerNames{k});
			[mu, invSigma, D] = buildGlobalStats(F, 1e-3);
			s = mdScores1(F, mu, invSigma);
			layerStats{k} = struct('LayerName',  layerNames{k}, ...
			                       'FeatureDim', D, ...
			                       'Mu',         mu, ...
			                       'InvSigma',   invSigma);
			scoreList{k}  = s;
		end

		S       = cat(2, scoreList{:});   % N × numL
		normMu  = mean(S, 1);
		normSig = std(S, 0, 1);
		normSig(normSig < 1e-8) = 1;

		branchScore = sum((S - normMu) ./ normSig, 2);
		bm = struct('LayerStats',  {layerStats}, ...
		            'NormMu',      normMu, ...
		            'NormSig',     normSig, ...
		            'Threshold',   prctile(branchScore, 97.5), ...
		            'TrainScores', branchScore);
		branchModels{b} = bm;
	end

	model = struct('NetworkID',    networkID, ...
	               'Algorithm',    'MBM', ...
	               'NumBranches',  numBranches, ...
	               'BranchModels', {branchModels});
end

function results = testMBM(net, testImds, model)
	nb      = model.NumBranches;
	N       = numel(testImds.Files);
	bScores = zeros(N, nb);
	bOOD    = false(N, nb);

	for b = 1:nb
		bm   = model.BranchModels{b};
		numL = numel(bm.LayerStats);
		S    = zeros(N, numL);

		for k = 1:numL
			ls    = bm.LayerStats{k};
			F     = feats(net, testImds, ls.LayerName);
			checkDim(F, ls.FeatureDim, ls.LayerName);
			S(:,k) = mdScores1(F, ls.Mu, ls.InvSigma);
		end

		bScores(:,b) = sum((S - bm.NormMu) ./ bm.NormSig, 2);
		bOOD(:,b)    = bScores(:,b) > bm.Threshold;
	end

	isOOD      = any(bOOD, 2);
	thresholds = cellfun(@(bm) bm.Threshold, model.BranchModels);

	results = struct('Algorithm',     'MBM', ...
	                 'IsOOD',         isOOD, ...
	                 'Accepted',      ~isOOD, ...
	                 'AcceptedCount', sum(~isOOD), ...
	                 'RejectedCount', sum(isOOD), ...
	                 'Scores',        bScores, ...
	                 'BranchIsOOD',   bOOD, ...
	                 'Threshold',     thresholds);
end

% =========================================================================
% SHARED HELPERS
% =========================================================================

function [mu, invSigma, D] = buildGlobalStats(F, lambda)
	% One-class pooled statistics: global mean + regularised covariance.
	N        = size(F, 1);
	D        = size(F, 2);
	mu       = mean(F, 1);
	Dc       = F - mu;
	Sigma    = (Dc' * Dc) / N + lambda * eye(D);
	invSigma = pinv(Sigma);
end

function scores = mdScores1(F, mu, invSigma)
	% One-class Mahalanobis distance squared (vectorised).
	d      = F - mu;
	scores = sum((d * invSigma) .* d, 2);
end

function F = feats(net, imds, layerName)
	% Extract activations and return N×D matrix.
	%
	% Streams imds in mini-batches and applies GAP immediately so the
	% running buffer is always N×C, never H×W×C×N.  Passing the full
	% datastore to activations() would pre-allocate the entire output
	% tensor (e.g. 320×390×16×38527 ≈ 286 GB for relu1) and crash.
	%
	% For CNN_chex conv layers (H×W×C×bN): GAP → N×C.
	% For MLP_chex FC layers   (1×1×D×bN): GAP is identity → N×D.
	BATCH  = 32;
	reset(imds);
	files  = imds.Files;
	N      = numel(files);
	readFn = imds.ReadFcn;
	F      = [];   % sized on first batch

	i = 1;
	while i <= N
		bEnd  = min(i + BATCH - 1, N);
		count = bEnd - i + 1;

		% Read mini-batch from disk
		imgs = cell(count, 1);
		for k = 1:count
			imgs{k} = readFn(files{i + k - 1});
		end
		imgBatch = cat(4, imgs{:});   % H×W×C×count

		% Activations on this small batch only
		A = double(activations(net, imgBatch, layerName));
		if ndims(A) == 4
			A = squeeze(mean(mean(A, 1), 2));   % C×count  or  C (count=1)
			if isvector(A), A = reshape(A, 1, []); else, A = A'; end
		end

		if isempty(F)
			F = zeros(N, size(A, 2), 'double');
		end
		F(i:bEnd, :) = A;
		i = bEnd + 1;
	end
end

function [F, fs] = fusedFeats(net, imds, layerNames, fsIn)
	% Build or apply per-layer z-score fusion stats then concatenate.
	numL   = numel(layerNames);
	blocks = cell(numL, 1);
	if isempty(fsIn)
		fs = repmat(struct('mu', [], 'sigma', []), numL, 1);
	else
		fs = fsIn;
	end
	for k = 1:numL
		A = feats(net, imds, layerNames{k});
		if isempty(fsIn)
			fs(k).mu    = mean(A, 1);
			fs(k).sigma = std(A, 0, 1);
		end
		sig = fs(k).sigma;
		sig(sig < 1e-8) = 1;
		blocks{k} = (A - fs(k).mu) ./ sig;
	end
	F = cat(2, blocks{:});
end

function checkDim(F, expected, name)
	if size(F, 2) ~= expected
		error('MD3_chex:dimMismatch', ...
			'Layer ''%s'': expected %d-dim features, got %d. Wrong network or model.', ...
			name, expected, size(F, 2));
	end
end

function results = packResults(algo, scores, isOOD, threshold)
	results = struct('Algorithm',     algo, ...
	                 'Scores',        scores, ...
	                 'Threshold',     threshold, ...
	                 'IsOOD',         isOOD, ...
	                 'Accepted',      ~isOOD, ...
	                 'AcceptedCount', sum(~isOOD), ...
	                 'RejectedCount', sum(isOOD));
end

function imds = toImds(testInput)
	if isa(testInput, 'matlab.io.datastore.ImageDatastore')
		imds = testInput;
		reset(imds);
	elseif iscell(testInput)
		imds = imageDatastore(testInput, 'ReadFcn', @readAndPreprocess);
	else
		error('MD3_chex:badTestInput', ...
			'testImds must be an imageDatastore or cell array of file paths.');
	end
end

function lc = autoLayerConfig(net, algorithm)
	names = {};
	for i = 1:numel(net.Layers)
		if isa(net.Layers(i), 'nnet.cnn.layer.ReLULayer')
			names{end+1} = net.Layers(i).Name; %#ok<AGROW>
		end
	end
	if isempty(names)
		error('MD3_chex:noReLU', 'No ReLU layers found. Provide layerConfig explicitly.');
	end
	switch algorithm
		case 'LHL'
			lc = names{end};
		case 'FUSION'
			lc = names;
		case 'MBM'
			lc = cellfun(@(n) {n}, names, 'UniformOutput', false);
	end
end

function cacheFile = getCacheFile(networkID, algorithm)
	here     = fileparts(mfilename('fullpath'));
	cacheDir = fullfile(here, 'trained_models');
	if ~isfolder(cacheDir), mkdir(cacheDir); end
	safeName  = regexprep(networkID, '[^a-zA-Z0-9_]', '_');
	cacheFile = fullfile(cacheDir, ...
		sprintf('stage3_chex_%s_%s.mat', safeName, lower(algorithm)));
end

function img = readAndPreprocess(filename)
	img = imread(filename);
	if size(img, 3) == 3
		img = rgb2gray(img);
	end
	img = im2single(img);
	if ismatrix(img)
		img = reshape(img, size(img, 1), size(img, 2), 1);
	end
end
