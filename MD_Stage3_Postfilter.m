function result = MD_Stage3_Postfilter(mode, net, varargin)
% MD_Stage3_Postfilter  Latent-space Mahalanobis OOD post-filter (Stage 3).
%
%   Implements three algorithms from Anthony & Kamnitsas (2023):
%
%     'LHL'    Algorithm 1 — Single layer (last hidden layer).
%              Per-class means, pooled covariance, min-class MD score.
%
%     'fusion' Algorithm 2 — Multi-layer feature concatenation.
%              Each layer z-score normalised then concatenated into one
%              vector; single pooled MD on the fused vector.
%
%     'MBM'    Algorithm 3 — Multi-Branch Mahalanobis.
%              Separate per-layer MD scores, normalised and summed within
%              each branch; one independent OOD detector per branch.
%              Sample is flagged OOD if ANY branch fires.
%
%   All covariance computation uses a pooled (tied) Sigma shared across
%   classes, with Tikhonov regularisation and pseudo-inverse for stability.
%
%   ── TRAIN ──────────────────────────────────────────────────────────────
%   model = MD_Stage3_Postfilter('train', net, networkID)
%   model = MD_Stage3_Postfilter('train', net, networkID, algorithm)
%   model = MD_Stage3_Postfilter('train', net, networkID, algorithm, layerConfig)
%   model = MD_Stage3_Postfilter('train', net, networkID, algorithm, layerConfig, forceRetrain)
%
%   Training data is loaded automatically from the standard MNIST path
%   (via getSetFolderPaths). The built manifold is cached per networkID
%   and algorithm so retraining is skipped on subsequent calls.
%
%   ── TEST ───────────────────────────────────────────────────────────────
%   results = MD_Stage3_Postfilter('test', net, XTest, model)
%   results = MD_Stage3_Postfilter('test', net, XTest, model, threshold)
%
%   XTest  : N×784 feature matrix (MLP / featureInputLayer)
%            or 28×28×1×N image array (CNN / imageInputLayer)
%
%   ── layerConfig ────────────────────────────────────────────────────────
%   'LHL'    : char, e.g. 'relu3'            (single layer name)
%   'fusion' : cellstr, e.g. {'relu1','relu2','relu3'}
%   'MBM'    : cell of cellstr (one cell per branch),
%              e.g. {{'relu1'},{'relu2'},{'relu3'}}
%   Omit layerConfig to auto-detect from the network's ReLU layers.
%
%   ── results struct fields ──────────────────────────────────────────────
%   .Algorithm      string
%   .IsOOD          N×1 logical
%   .Accepted       N×1 logical
%   .AcceptedCount  scalar
%   .RejectedCount  scalar
%   .Scores         N×1  (LHL / fusion)  or  N×numBranches  (MBM)
%   .Threshold      scalar (LHL / fusion) or numBranches×1 (MBM)

	switch lower(mode)
		case 'train'
			result = doTrain(net, varargin{:});
		case 'test'
			result = doTest(net, varargin{:});
		otherwise
			error('MD_Stage3_Postfilter:badMode', ...
				'mode must be ''train'' or ''test''.');
	end
end

% =========================================================================
% TRAIN DISPATCH
% =========================================================================

function model = doTrain(net, networkID, algorithm, layerConfig, forceRetrain)
	if nargin < 2 || isempty(networkID)
		error('MD_Stage3_Postfilter:noID', 'networkID is required.');
	end
	if nargin < 3 || isempty(algorithm),    algorithm    = 'LHL';   end
	if nargin < 5 || isempty(forceRetrain), forceRetrain = false;   end

	algorithm = upper(algorithm);

	cacheFile = getCacheFile(networkID, algorithm);
	if ~forceRetrain && isfile(cacheFile)
		data = load(cacheFile, 'model');
		if isfield(data, 'model') && ...
				strcmp(data.model.NetworkID, networkID) && ...
				strcmp(data.model.Algorithm, algorithm)
			model = data.model;
			fprintf('Loaded Stage3 cache  [%s | %s]\n', algorithm, networkID);
			return;
		end
	end

	if nargin < 4 || isempty(layerConfig)
		layerConfig = autoLayerConfig(net, algorithm);
	end

	[XTrain, YInt, numClasses] = loadTrainData(net);

	fprintf('Building Stage3 manifold  [%s | %s]...\n', algorithm, networkID);

	switch algorithm
		case 'LHL'
			model = trainLHL(net, XTrain, YInt, numClasses, layerConfig, networkID);
		case 'FUSION'
			model = trainFusion(net, XTrain, YInt, numClasses, layerConfig, networkID);
		case 'MBM'
			model = trainMBM(net, XTrain, YInt, numClasses, layerConfig, networkID);
		otherwise
			error('MD_Stage3_Postfilter:badAlgo', ...
				'algorithm must be ''LHL'', ''fusion'', or ''MBM''.');
	end

	save(cacheFile, 'model', '-v7.3');
	fprintf('Saved Stage3 cache to: %s\n', cacheFile);
end

% =========================================================================
% TEST DISPATCH
% =========================================================================

function results = doTest(net, XTest, model, threshold)
	if nargin < 4, threshold = []; end

	switch model.Algorithm
		case 'LHL'
			results = testLHL(net, XTest, model, threshold);
		case 'FUSION'
			results = testFusion(net, XTest, model, threshold);
		case 'MBM'
			results = testMBM(net, XTest, model);
	end

	fprintf('Stage3 [%s]  Accepted: %d  Rejected: %d  (threshold: %s)\n', ...
		results.Algorithm, results.AcceptedCount, results.RejectedCount, ...
		mat2str(results.Threshold, 4));
end

% =========================================================================
% ALGORITHM 1 — LHL
% =========================================================================

function model = trainLHL(net, XTrain, YInt, numClasses, layerName, networkID)
	fprintf('  LHL: extracting layer ''%s''\n', layerName);
	F = feats(net, XTrain, layerName);
	[mu, invSigma, D] = buildPooledStats(F, YInt, numClasses, 1e-3);

	scores    = mdScores(F, mu, invSigma, numClasses);
	threshold = prctile(scores, 97.5);

	model = struct('NetworkID',   networkID,  'Algorithm',  'LHL', ...
	               'LayerName',   layerName,  'FeatureDim', D, ...
	               'Mu',          mu,         'InvSigma',   invSigma, ...
	               'Threshold',   threshold,  'TrainScores', scores);
end

function results = testLHL(net, XTest, model, threshold)
	if isempty(threshold), threshold = model.Threshold; end
	F = feats(net, XTest, model.LayerName);
	checkDim(F, model.FeatureDim, model.LayerName);
	scores = mdScores(F, model.Mu, model.InvSigma, size(model.Mu, 1));
	results = packResults('LHL', scores, scores > threshold, threshold);
end

% =========================================================================
% ALGORITHM 2 — FUSION
% =========================================================================

function model = trainFusion(net, XTrain, YInt, numClasses, layerNames, networkID)
	fprintf('  Fusion: layers  {%s}\n', strjoin(layerNames, ', '));
	[F, fusionStats] = fusedFeats(net, XTrain, layerNames, []);
	[mu, invSigma, D] = buildPooledStats(F, YInt, numClasses, 1e-2);

	scores    = mdScores(F, mu, invSigma, numClasses);
	threshold = prctile(scores, 97.5);

	model = struct('NetworkID',   networkID,   'Algorithm',    'FUSION', ...
	               'LayerNames',  {layerNames}, 'FusionStats',  fusionStats, ...
	               'FeatureDim',  D,            'Mu',           mu, ...
	               'InvSigma',   invSigma,     'Threshold',    threshold, ...
	               'TrainScores', scores);
end

function results = testFusion(net, XTest, model, threshold)
	if isempty(threshold), threshold = model.Threshold; end
	F = fusedFeats(net, XTest, model.LayerNames, model.FusionStats);
	checkDim(F, model.FeatureDim, 'fused');
	scores = mdScores(F, model.Mu, model.InvSigma, size(model.Mu, 1));
	results = packResults('FUSION', scores, scores > threshold, threshold);
end

% =========================================================================
% ALGORITHM 3 — MBM
% =========================================================================

function model = trainMBM(net, XTrain, YInt, numClasses, branches, networkID)
	numBranches  = numel(branches);
	branchModels = cell(numBranches, 1);

	for b = 1:numBranches
		layerNames = branches{b};
		numL       = numel(layerNames);
		fprintf('  MBM branch %d/%d: {%s}\n', b, numBranches, strjoin(layerNames, ', '));

		layerStats  = cell(numL, 1);
		scoreList   = cell(numL, 1);

		for k = 1:numL
			F = feats(net, XTrain, layerNames{k});
			[mu, invSigma, D] = buildPooledStats(F, YInt, numClasses, 1e-3);
			s = mdScores(F, mu, invSigma, numClasses);

			layerStats{k} = struct('LayerName',  layerNames{k}, ...
			                       'FeatureDim', D, ...
			                       'Mu',         mu, ...
			                       'InvSigma',   invSigma);
			scoreList{k}  = s;
		end

		% N × numL matrix of per-layer training scores
		S = cat(2, scoreList{:});

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

function results = testMBM(net, XTest, model)
	nb   = model.NumBranches;
	N    = numSamp(XTest);
	bScores = zeros(N, nb);
	bOOD    = false(N, nb);

	for b = 1:nb
		bm   = model.BranchModels{b};
		numL = numel(bm.LayerStats);
		S    = zeros(N, numL);

		for k = 1:numL
			ls = bm.LayerStats{k};
			F  = feats(net, XTest, ls.LayerName);
			checkDim(F, ls.FeatureDim, ls.LayerName);
			S(:, k) = mdScores(F, ls.Mu, ls.InvSigma, size(ls.Mu, 1));
		end

		bScores(:, b) = sum((S - bm.NormMu) ./ bm.NormSig, 2);
		bOOD(:, b)    = bScores(:, b) > bm.Threshold;
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

function [mu, invSigma, D] = buildPooledStats(F, YInt, numClasses, lambda)
	N  = size(F, 1);
	D  = size(F, 2);
	mu = zeros(numClasses, D);
	for c = 0:numClasses-1
		idx        = (YInt == c);
		mu(c+1, :) = mean(F(idx, :), 1);
	end
	Sigma = zeros(D, D);
	for c = 0:numClasses-1
		idx  = (YInt == c);
		Dc   = F(idx, :) - mu(c+1, :);
		Sigma = Sigma + Dc' * Dc;
	end
	Sigma    = Sigma / N + lambda * eye(D);
	invSigma = pinv(Sigma);
end

function scores = mdScores(F, mu, invSigma, numClasses)
	% Vectorised: avoids per-sample loop.
	N      = size(F, 1);
	allD   = zeros(N, numClasses);
	for c = 1:numClasses
		d         = F - mu(c, :);          % N × D
		allD(:,c) = sum((d * invSigma) .* d, 2);  % N × 1
	end
	scores = min(allD, [], 2);
end

function F = feats(net, X, layerName)
	% Extract activations from one layer.
	% Conv output (H×W×C×N): global average pooling → N×C.
	% FC output  (1×1×D×N): squeeze+transpose     → N×D.
	A = activations(net, X, layerName);
	A = squeeze(mean(mean(double(A), 1), 2));  % C×N  (works for both)
	if isvector(A)
		F = reshape(A, 1, []);   % single sample → 1×C
	else
		F = A';                  % N×C
	end
end

function [F, fs] = fusedFeats(net, X, layerNames, fsIn)
	% Build or apply per-layer z-score fusion stats, then concatenate.
	numL   = numel(layerNames);
	blocks = cell(numL, 1);
	if isempty(fsIn)
		fs = repmat(struct('mu', [], 'sigma', []), numL, 1);
	else
		fs = fsIn;
	end
	for k = 1:numL
		A = feats(net, X, layerNames{k});
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
		error('MD_Stage3_Postfilter:dimMismatch', ...
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

function N = numSamp(X)
	if ndims(X) == 4
		N = size(X, 4);   % image batch H×W×C×N
	else
		N = size(X, 1);   % feature matrix N×D
	end
end

function lc = autoLayerConfig(net, algorithm)
	% Collect all ReLU layer names in order.
	names = {};
	for i = 1:numel(net.Layers)
		if isa(net.Layers(i), 'nnet.cnn.layer.ReLULayer')
			names{end+1} = net.Layers(i).Name; %#ok<AGROW>
		end
	end
	if isempty(names)
		error('MD_Stage3_Postfilter:noReLU', ...
			'No ReLU layers found in network. Provide layerConfig explicitly.');
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

% =========================================================================
% DATA LOADING  (mirrors MD_Stage1_Prefilter / MLP_reader pattern)
% =========================================================================

function [XTrain, YInt, numClasses] = loadTrainData(net)
	trainRoot = getSetFolderPaths('resolve', 'trainRoot');
	imPath    = fullfile(trainRoot, 'train-images-idx3-ubyte');
	lblPath   = fullfile(trainRoot, 'train-labels-idx1-ubyte');

	if ~isfile(imPath) || ~isfile(lblPath)
		error('MD_Stage3_Postfilter:missingData', ...
			'Training IDX files not found in: %s', trainRoot);
	end

	fprintf('  Loading training data from: %s\n', trainRoot);

	% Detect input format expected by the network
	inputLayer = net.Layers(1);
	if isa(inputLayer, 'nnet.cnn.layer.ImageInputLayer')
		XTrain = readIDXImages4D(imPath);   % 28×28×1×N
	else
		XTrain = readIDXFeatures(imPath);   % N×784
	end

	labels    = readIDXLabels(lblPath);
	YInt      = double(labels(:));
	numClasses = numel(unique(YInt));
end

function images = readIDXImages4D(filePath)
	fid     = fopen(filePath, 'rb');
	cleaner = onCleanup(@() fclose(fid));
	fread(fid, 1, 'int32', 0, 'ieee-be');          % magic
	N    = fread(fid, 1, 'int32', 0, 'ieee-be');
	rows = fread(fid, 1, 'int32', 0, 'ieee-be');
	cols = fread(fid, 1, 'int32', 0, 'ieee-be');
	px   = fread(fid, N*rows*cols, 'uint8=>single');
	im3  = reshape(px, [cols, rows, N]);
	im3  = permute(im3, [2, 1, 3]) ./ 255;
	images = reshape(im3, [rows, cols, 1, N]);
end

function X = readIDXFeatures(filePath)
	fid     = fopen(filePath, 'rb');
	cleaner = onCleanup(@() fclose(fid));
	fread(fid, 1, 'int32', 0, 'ieee-be');
	N    = fread(fid, 1, 'int32', 0, 'ieee-be');
	rows = fread(fid, 1, 'int32', 0, 'ieee-be');
	cols = fread(fid, 1, 'int32', 0, 'ieee-be');
	px   = fread(fid, N*rows*cols, 'uint8=>single');
	X    = reshape(px, [rows*cols, N])' ./ 255;   % N×784
end

function labels = readIDXLabels(filePath)
	fid     = fopen(filePath, 'rb');
	cleaner = onCleanup(@() fclose(fid));
	fread(fid, 1, 'int32', 0, 'ieee-be');
	N      = fread(fid, 1, 'int32', 0, 'ieee-be');
	labels = fread(fid, N, 'uint8=>double');
end

function cacheFile = getCacheFile(networkID, algorithm)
	here     = fileparts(mfilename('fullpath'));
	cacheDir = fullfile(here, 'trained_models');
	if ~isfolder(cacheDir), mkdir(cacheDir); end
	safeName  = regexprep(networkID, '[^a-zA-Z0-9_]', '_');
	cacheFile = fullfile(cacheDir, ...
		sprintf('stage3_%s_%s.mat', safeName, lower(algorithm)));
end
