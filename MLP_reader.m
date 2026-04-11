function results = MLP_reader(dataRoot, forceRetrain)
% MLP_reader  Train and evaluate a fully-connected MLP on raw MNIST IDX files.
%
%   RESULTS = MLP_reader() reads files from .\MNIST\raw relative to this
%   function file, trains the MLP, tests it, and returns a struct with
%   predicted labels, true labels, confusion matrix, confusion table, and
%   accuracy.
%
%   RESULTS = MLP_reader(dataRoot) uses the provided MNIST raw folder path.
%
%   RESULTS = MLP_reader(dataRoot, forceRetrain) retrains when forceRetrain
%   is true. Otherwise, the function reuses the last saved trained MLP when
%   both its training-data root and hidden-layer configuration match.
%
%   Output struct fields match CNN_reader:
%     .Network          – trained SeriesNetwork object
%     .YTrue            – categorical test labels
%     .YPred            – categorical predicted labels
%     .ConfusionMatrix  – 10×10 numeric confusion matrix
%     .ConfusionTable   – table version of the confusion matrix
%     .Accuracy         – scalar accuracy in percent

	% ====================================================================
	% ARCHITECTURE CONFIGURATION  — edit here
	%   hiddenLayerSizes : row vector of hidden-layer widths (1–1024 each).
	%     0 elements  →  linear classifier (input → output directly)
	%     1–5 elements →  one hidden layer per element
	%   Examples:
	%     []                → no hidden layers
	%     [256]             → single hidden layer of 256
	%     [512, 256, 128]   → three hidden layers  ← default
	%     [1024,512,256,128,64] → five hidden layers
	% ====================================================================
	hiddenLayerSizes = [512, 256, 128];
	% ====================================================================

	% Validate architecture
	if numel(hiddenLayerSizes) > 5
		error('MLP_reader:tooManyLayers', ...
			'At most 5 hidden layers are supported (got %d).', numel(hiddenLayerSizes));
	end
	if ~isempty(hiddenLayerSizes)
		if any(hiddenLayerSizes < 1) || any(hiddenLayerSizes > 1024)
			error('MLP_reader:invalidWidth', ...
				'Each hidden layer must have between 1 and 1024 neurons.');
		end
	end

	if nargin < 1
		dataRoot = '';
	end
	dataRoot = getSetFolderPaths('resolve', 'trainRoot', dataRoot);
	if nargin < 2
		forceRetrain = false;
	end
	classNames = getDatasetClassNames(dataRoot);

	trainImagesPath = fullfile(dataRoot, 'train-images-idx3-ubyte');
	trainLabelsPath = fullfile(dataRoot, 'train-labels-idx1-ubyte');
	testImagesPath  = fullfile(dataRoot, 't10k-images-idx3-ubyte');
	testLabelsPath  = fullfile(dataRoot, 't10k-labels-idx1-ubyte');

	validateMNISTFiles(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath);

	fprintf('loading training data from: %s\n', getSetFolderPaths(dataRoot));
	[XTrain4D, YTrain] = loadMNIST(trainImagesPath, trainLabelsPath, classNames);
	[XTest4D,  YTest]  = loadMNIST(testImagesPath,  testLabelsPath, classNames);

	% Flatten 28×28×1×N images → N×784 feature matrices for featureInputLayer
	XTrain = flattenImages(XTrain4D);
	XTest  = flattenImages(XTest4D);

	% Build layer array from configuration
	layers = buildLayers(hiddenLayerSizes);

	options = trainingOptions('adam', ...
		'InitialLearnRate', 1e-3, ...
		'MaxEpochs', 10, ...
		'MiniBatchSize', 256, ...
		'Shuffle', 'every-epoch', ...
		'ValidationData', {XTest, YTest}, ...
		'ValidationFrequency', 100, ...
		'Verbose', true, ...
		'Plots', 'training-progress');

	if isempty(hiddenLayerSizes)
		archStr = 'no hidden layers';
	else
		archStr = strjoin(string(hiddenLayerSizes), '-');
	end

	cacheFile = getMLPCacheFile();
	[net, loadedFromCache] = tryLoadMLPCache(cacheFile, trainImagesPath, trainLabelsPath, hiddenLayerSizes, forceRetrain);
	if loadedFromCache
		fprintf('Loaded cached MLP model from: %s\n', getSetFolderPaths(cacheFile));
	else
		fprintf('Training MLP  784 → [%s] → 10 ...\n', archStr);
		net = trainNetwork(XTrain, YTrain, layers, options);
		saveMLPCache(cacheFile, net, trainImagesPath, trainLabelsPath, hiddenLayerSizes);
		fprintf('Saved MLP cache to: %s\n', getSetFolderPaths(cacheFile));
	end

	fprintf('Running inference on test set...\n');
	YPred = classify(net, XTest, 'MiniBatchSize', 512);

	accuracy = mean(YPred == YTest) * 100;
	confMat  = confusionmat(YTest, YPred);
	confTbl = array2table(confMat, ...
		'VariableNames', cellstr(matlab.lang.makeValidName("Pred_" + classNames)), ...
		'RowNames',      cellstr(matlab.lang.makeValidName("True_" + classNames)));

	fprintf('\nTest Accuracy: %.2f%%\n', accuracy);
	disp('Confusion Table (rows=True, cols=Pred):');
	disp(confTbl);

	figure('Name', 'MNIST MLP Confusion Matrix', 'Color', 'w');
	confusionchart(YTest, YPred);
	title(sprintf('MNIST MLP Confusion Matrix  [%s]  (Accuracy: %.2f%%)', archStr, accuracy));

	results = struct();
	results.Network         = net;
	results.YTrue           = YTest;
	results.YPred           = YPred;
	results.ConfusionMatrix = confMat;
	results.ConfusionTable  = confTbl;
	results.Accuracy        = accuracy;
	results.ClassNames      = classNames;
end

function cacheFile = getMLPCacheFile()
	here = fileparts(mfilename('fullpath'));
	cacheDir = fullfile(here, 'trained_models');
	if ~isfolder(cacheDir)
		mkdir(cacheDir);
	end
	cacheFile = fullfile(cacheDir, 'mlp_reader_cache.mat');
end

function [net, loadedFromCache] = tryLoadMLPCache(cacheFile, trainImagesPath, trainLabelsPath, hiddenLayerSizes, forceRetrain)
	net = [];
	loadedFromCache = false;
	if forceRetrain || ~isfile(cacheFile)
		return;
	end

	data = load(cacheFile, 'net', 'cacheMeta');
	if ~isfield(data, 'net') || ~isfield(data, 'cacheMeta')
		return;
	end
	if ~isfield(data.cacheMeta, 'trainImagesPath') || ~isfield(data.cacheMeta, 'trainLabelsPath') || ...
			~isfield(data.cacheMeta, 'hiddenLayerSizes')
		return;
	end
	if ~strcmp(data.cacheMeta.trainImagesPath, trainImagesPath) || ...
			~strcmp(data.cacheMeta.trainLabelsPath, trainLabelsPath)
		return;
	end
	if ~isequal(data.cacheMeta.hiddenLayerSizes, hiddenLayerSizes)
		return;
	end

	net = data.net;
	loadedFromCache = true;
end

function saveMLPCache(cacheFile, net, trainImagesPath, trainLabelsPath, hiddenLayerSizes)
	cacheMeta = struct();
	cacheMeta.trainImagesPath = trainImagesPath;
	cacheMeta.trainLabelsPath = trainLabelsPath;
	cacheMeta.hiddenLayerSizes = hiddenLayerSizes;
	cacheMeta.savedAt = char(datetime('now'));
	save(cacheFile, 'net', 'cacheMeta', '-v7.3');
end

% ======================================================================
% Build MLP layer array from a vector of hidden-layer sizes.
% Input layer  : featureInputLayer(784)
% Hidden layers: fullyConnectedLayer → reluLayer  (repeated)
% Output layer : fullyConnectedLayer(10) → softmaxLayer → classificationLayer
% ======================================================================
function layers = buildLayers(hiddenSizes)
	numInputFeatures = 784;   % 28×28
	numClasses       = 10;    % digits 0–9

	layers = featureInputLayer(numInputFeatures, 'Name', 'input', ...
		'Normalization', 'none');

	for k = 1:numel(hiddenSizes)
		layers = [layers
			fullyConnectedLayer(hiddenSizes(k), 'Name', sprintf('fc%d',   k))
			reluLayer(                           'Name', sprintf('relu%d', k))]; %#ok<AGROW>
	end

	outIdx = numel(hiddenSizes) + 1;
	layers = [layers
		fullyConnectedLayer(numClasses, 'Name', sprintf('fc%d', outIdx))
		softmaxLayer(                   'Name', 'softmax')
		classificationLayer(            'Name', 'output')];
end

% ======================================================================
% Reshape 4-D image array (H×W×C×N) to feature matrix (N×HWC).
% ======================================================================
function X = flattenImages(images4D)
	[H, W, C, N] = size(images4D);
	X = reshape(images4D, H*W*C, N)';   % N × 784
end

% ======================================================================
% Helper functions – identical to CNN_reader.m
% ======================================================================
function validateMNISTFiles(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath)
	required = {trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath};
	for i = 1:numel(required)
		if ~isfile(required{i})
			error('Missing required file: %s', required{i});
		end
	end
end

function [images4D, labelsCat] = loadMNIST(imagesPath, labelsPath, classNames)
	images = readIDXImages(imagesPath);
	labels = readIDXLabels(labelsPath);

	if size(images, 4) ~= numel(labels)
		error('Image/label count mismatch in %s and %s', imagesPath, labelsPath);
	end

	images4D   = images;
	labelsCat  = categorical(labels, 0:9, classNames);
end

function classNames = getDatasetClassNames(dataRoot)
	if contains(lower(char(string(dataRoot))), 'fashion')
		classNames = ["T-shirt_top", "Trouser", "Pullover", "Dress", "Coat", ...
			"Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"];
	else
		classNames = string(0:9);
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
	numRows   = fread(fid, 1, 'int32', 0, 'ieee-be');
	numCols   = fread(fid, 1, 'int32', 0, 'ieee-be');

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
