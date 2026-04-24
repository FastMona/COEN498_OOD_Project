function results = MD_Stage1_Prefilter(testInput, vigilance, forceRetrain, verboseOutput)
% MD_Stage1_Prefilter Train or apply a digit-manifold OOD detector.
%
%   MD_Stage1_Prefilter() trains or loads the manifold model built from MNIST_digits
%   and remembers it on disk. No test data is evaluated in this mode.
%
%   RESULTS = MD_Stage1_Prefilter(testInput) evaluates one sample, a batch of samples,
%   or a folder of IDX test patterns using the cached manifold model.
%
%   RESULTS = MD_Stage1_Prefilter(testInput, vigilance) changes the acceptance
%   threshold. Default vigilance is 0.5.
%
%   RESULTS = MD_Stage1_Prefilter(testInput, vigilance, forceRetrain) forces manifold
%   retraining when true.
%
%   RESULTS = MD_Stage1_Prefilter(testInput, vigilance, forceRetrain, verboseOutput)
%   enables or suppresses console output.
%
%   testInput may be one of:
%     - folder path containing t10k-*.idx files
%     - single 28x28 image
%     - single 1x784 feature vector
%     - batch Nx784 feature matrix
%     - 28x28xN image array
%     - 28x28x1xN image array

	if nargin < 2 || isempty(vigilance)
		vigilance = 0.5;
	end
	if nargin < 3 || isempty(forceRetrain)
		forceRetrain = false;
	end
	if nargin < 4 || isempty(verboseOutput)
		verboseOutput = true;
	end
	if vigilance < 0 || vigilance > 1
		error('MD_Stage1_Prefilter:badThreshold', 'vigilance must be between 0 and 1.');
	end

	trainRoot = getSetFolderPaths('resolve', 'trainRoot');
	if nargin >= 1 && (ischar(testInput) || (isstring(testInput) && isscalar(testInput)))
		testInput = getSetFolderPaths('resolve', 'testRoot', char(string(testInput)));
	end
	model = loadOrTrainModel(trainRoot, forceRetrain, verboseOutput);

	if nargin < 1 || isempty(testInput)
		if verboseOutput
			fprintf('\nMD_Stage1_Prefilter model ready\n');
			fprintf('Train data: %s\n', getSetFolderPaths(trainRoot));
			fprintf('Digits modeled: 0..9\n');
		end
		results = struct();
		results.Mode = 'train-only';
		results.TrainDataRoot = trainRoot;
		results.Model = model;
		results.Vigilance = vigilance;
		return;
	end

	[XTest, images4D, inputLabels, inputSource] = normalizeTestInput(testInput);
	[bestDigit, bestDistance, confidence, accepted, distanceMatrix, confidenceMatrix] = ...
		scoreSamples(model, XTest, vigilance);

	numSamples = size(XTest, 1);
	acceptedCount = sum(accepted);
	rejectedCount = numSamples - acceptedCount;
	if acceptedCount > 0
		acceptedDigits = bestDigit(accepted);
		acceptedDigitCounts = histcounts(acceptedDigits, -0.5:1:9.5);
	else
		acceptedDigitCounts = zeros(1, 10);
	end

	if verboseOutput
		fprintf('\nMD_Stage1_Prefilter summary\n');
		fprintf('Running inference test on: %s\n', getSetFolderPaths(inputSource));
		fprintf('Vigilance:  %.2f\n', vigilance);
		fprintf('Samples:    %d\n', numSamples);
		fprintf('Accepted:   %d (%.2f%%)\n', acceptedCount, 100 * acceptedCount / numSamples);
		fprintf('Rejected:   %d (%.2f%%)\n', rejectedCount, 100 * rejectedCount / numSamples);
		if acceptedCount > 0
			fprintf('Accepted digits: ');
			for digit = 0:9
				fprintf('%d:%d', digit, acceptedDigitCounts(digit + 1));
				if digit < 9
					fprintf('  ');
				else
					fprintf('\n');
				end
			end
		else
			fprintf('Accepted digits: none\n');
		end
	end

	results = struct();
	results.Mode = 'score';
	results.TrainDataRoot = trainRoot;
	results.InputSource = inputSource;
	results.Vigilance = vigilance;
	results.Model = model;
	results.Features = XTest;
	results.Images4D = images4D;
	results.InputLabels = inputLabels;
	results.BestDigit = bestDigit;
	results.BestDistance = bestDistance;
	results.Confidence = confidence;
	results.Accepted = accepted;
	results.IsOOD = ~accepted;
	results.AcceptedCount = acceptedCount;
	results.RejectedCount = rejectedCount;
	results.AcceptedDigitCounts = acceptedDigitCounts;
	results.DistanceMatrix = distanceMatrix;
	results.ConfidenceMatrix = confidenceMatrix;
end

function model = loadOrTrainModel(trainRoot, forceRetrain, verboseOutput)
	cacheFile = getCacheFile();
	if ~forceRetrain && isfile(cacheFile)
		data = load(cacheFile, 'model', 'cacheMeta');
		if isfield(data, 'model') && isfield(data, 'cacheMeta') && ...
				isfield(data.cacheMeta, 'trainRoot') && strcmp(data.cacheMeta.trainRoot, trainRoot)
			model = data.model;
			if verboseOutput
				fprintf('Loaded cached manifold model from: %s\n', getSetFolderPaths(cacheFile));
			end
			return;
		end
	end

	if verboseOutput
		fprintf('Training manifold model from: %s\n', getSetFolderPaths(trainRoot));
	end
	[XTrain, YTrain] = loadTrainFeatures(trainRoot);
	model = trainDigitManifolds(XTrain, YTrain);

	cacheMeta = struct();
	cacheMeta.trainRoot = trainRoot;
	cacheMeta.savedAt = char(datetime('now'));
	save(cacheFile, 'model', 'cacheMeta', '-v7.3');
	if verboseOutput
		fprintf('Saved manifold model to: %s\n', getSetFolderPaths(cacheFile));
	end
end

function cacheFile = getCacheFile()
	here = fileparts(mfilename('fullpath'));
	cacheDir = fullfile(here, 'trained_models');
	if ~isfolder(cacheDir)
		mkdir(cacheDir);
	end
	cacheFile = fullfile(cacheDir, 'md_filter_cache.mat');
end

function model = trainDigitManifolds(XTrain, YTrain)
	model = struct();
	model.Digits = 0:9;
	model.NumComponents = 20;
	model.Manifolds = repmat(struct( ...
		'digit', [], 'mu', [], 'basis', [], 'latent', [], ...
		'residualVar', [], 'distanceScale', []), 10, 1);

	for digit = 0:9
		mask = (YTrain == digit);
		XDigit = XTrain(mask, :);
		mu = mean(XDigit, 1);
		XCentered = XDigit - mu;

		[~, S, V] = svd(XCentered, 'econ');
		numComp = min(model.NumComponents, size(V, 2));
		basis = V(:, 1:numComp);
		singularValues = diag(S);
		latent = (singularValues(1:numComp) .^ 2) / max(size(XDigit, 1) - 1, 1);
		latent = max(latent, 1e-6);

		proj = XCentered * basis;
		recon = proj * basis';
		residual = XCentered - recon;
		residualEnergy = sum(residual .^ 2, 2) / size(XDigit, 2);
		residualVar = max(median(residualEnergy), 1e-6);

		trainDist = mahalanobisToManifold(XDigit, mu, basis, latent, residualVar);
		distanceScale = max(prctile(trainDist, 95), 1e-6);

		model.Manifolds(digit + 1).digit = digit;
		model.Manifolds(digit + 1).mu = mu;
		model.Manifolds(digit + 1).basis = basis;
		model.Manifolds(digit + 1).latent = latent;
		model.Manifolds(digit + 1).residualVar = residualVar;
		model.Manifolds(digit + 1).distanceScale = distanceScale;
	end
end

function [bestDigit, bestDistance, confidence, accepted, distanceMatrix, confidenceMatrix] = scoreSamples(model, XTest, vigilance)
	numSamples = size(XTest, 1);
	distanceMatrix = zeros(numSamples, 10);
	confidenceMatrix = zeros(numSamples, 10);

	for idx = 1:10
		manifold = model.Manifolds(idx);
		distanceMatrix(:, idx) = mahalanobisToManifold( ...
			XTest, manifold.mu, manifold.basis, manifold.latent, manifold.residualVar);
		normalizedDistance = distanceMatrix(:, idx) ./ max(manifold.distanceScale, 1e-6);
		confidenceMatrix(:, idx) = exp(-0.5 * normalizedDistance .^ 2);
	end

	[confidence, bestIdx] = max(confidenceMatrix, [], 2);
	bestDigit = bestIdx - 1;
	bestDistance = distanceMatrix(sub2ind(size(distanceMatrix), (1:numSamples)', bestIdx));
	accepted = confidence >= vigilance;
end

function distance = mahalanobisToManifold(X, mu, basis, latent, residualVar)
	centered = X - mu;
	proj = centered * basis;
	recon = proj * basis';
	residual = centered - recon;

	latent = reshape(max(latent, 1e-6), 1, []);
	subspaceTerm = sum((proj .^ 2) ./ latent, 2) / numel(latent);
	residualEnergy = sum(residual .^ 2, 2) / size(X, 2);
	residualTerm = residualEnergy / max(residualVar, 1e-6);
	distance = sqrt(subspaceTerm + residualTerm);
end

function [XTest, images4D, inputLabels, inputSource] = normalizeTestInput(testInput)
	images4D = [];
	inputLabels = [];

	if ischar(testInput) || (isstring(testInput) && isscalar(testInput))
		folderPath = char(string(testInput));
		[XTest, inputLabels, images4D] = loadTestFeatures(folderPath);
		inputSource = folderPath;
		return;
	end

	if isnumeric(testInput)
		XTest = featuresFromNumericInput(testInput);
		inputSource = sprintf('numeric input (%d sample(s))', size(XTest, 1));
		return;
	end

	error('MD_Stage1_Prefilter:unsupportedInput', ...
		'Unsupported testInput type. Use a folder path or numeric image/features.');
end

function X = featuresFromNumericInput(testInput)
	if isvector(testInput)
		if numel(testInput) ~= 784
			error('MD_Stage1_Prefilter:badVector', 'Feature vector input must have 784 elements.');
		end
		X = reshape(single(testInput), 1, 784);
		return;
	end

	inputSize = size(testInput);
	if ismatrix(testInput)
		if isequal(inputSize, [28, 28])
			X = reshape(single(testInput), 1, 784);
			return;
		end
		if inputSize(2) == 784
			X = single(testInput);
			return;
		end
	end

	if ndims(testInput) == 3 && inputSize(1) == 28 && inputSize(2) == 28
		X = reshape(single(testInput), 784, inputSize(3))';
		return;
	end

	if ndims(testInput) == 4 && inputSize(1) == 28 && inputSize(2) == 28 && inputSize(3) == 1
		X = reshape(single(testInput), 784, inputSize(4))';
		return;
	end

	error('MD_Stage1_Prefilter:badNumericInput', ...
		'Numeric input must be 28x28, Nx784, 28x28xN, or 28x28x1xN.');
end

function [XTrain, YTrain] = loadTrainFeatures(dataRoot)
	trainImagesPath = fullfile(dataRoot, 'train-images-idx3-ubyte');
	trainLabelsPath = fullfile(dataRoot, 'train-labels-idx1-ubyte');
	testImagesPath = fullfile(dataRoot, 't10k-images-idx3-ubyte');
	testLabelsPath = fullfile(dataRoot, 't10k-labels-idx1-ubyte');
	validateMNISTFiles(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath);

	images4D = readIDXImages(trainImagesPath);
	labels = readIDXLabels(trainLabelsPath);
	XTrain = flattenImages(images4D);
	YTrain = labels(:);
end

function [XTest, labels, images4D] = loadTestFeatures(dataRoot)
	testImagesPath = fullfile(dataRoot, 't10k-images-idx3-ubyte');
	testLabelsPath = fullfile(dataRoot, 't10k-labels-idx1-ubyte');
	if ~isfile(testImagesPath)
		error('MD_Stage1_Prefilter:missingOODImages', 'Missing file: %s', testImagesPath);
	end
	if ~isfile(testLabelsPath)
		error('MD_Stage1_Prefilter:missingOODLabels', 'Missing file: %s', testLabelsPath);
	end

	images4D = readIDXImages(testImagesPath);
	labels = readIDXLabels(testLabelsPath);
	XTest = flattenImages(images4D);
end

function X = flattenImages(images4D)
	[H, W, C, N] = size(images4D);
	X = reshape(images4D, H * W * C, N)';
end

function validateMNISTFiles(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath)
	required = {trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath};
	for i = 1:numel(required)
		if ~isfile(required{i})
			error('Missing required file: %s', required{i});
		end
	end
end

function images4D = readIDXImages(filePath)
	fid = fopen(filePath, 'rb');
	if fid < 0
		error('Could not open image file: %s', filePath);
	end
	cleaner = onCleanup(@() fclose(fid));

	magic = fread(fid, 1, 'int32', 0, 'ieee-be');
	if magic ~= 2051
		error('Invalid image IDX magic number in %s (got %d).', filePath, magic);
	end

	numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
	numRows = fread(fid, 1, 'int32', 0, 'ieee-be');
	numCols = fread(fid, 1, 'int32', 0, 'ieee-be');

	pixels = fread(fid, numImages * numRows * numCols, 'uint8=>single');
	if numel(pixels) ~= numImages * numRows * numCols
		error('Unexpected end of image file: %s', filePath);
	end

	images3D = reshape(pixels, [numCols, numRows, numImages]);
	images3D = permute(images3D, [2, 1, 3]);
	images3D = images3D ./ 255;
	images4D = reshape(images3D, [numRows, numCols, 1, numImages]);
end

function labels = readIDXLabels(filePath)
	fid = fopen(filePath, 'rb');
	if fid < 0
		error('Could not open label file: %s', filePath);
	end
	cleaner = onCleanup(@() fclose(fid));

	magic = fread(fid, 1, 'int32', 0, 'ieee-be');
	if magic ~= 2049
		error('Invalid label IDX magic number in %s (got %d).', filePath, magic);
	end

	numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
	labels = fread(fid, numLabels, 'uint8=>double');
	if numel(labels) ~= numLabels
		error('Unexpected end of label file: %s', filePath);
	end
end
