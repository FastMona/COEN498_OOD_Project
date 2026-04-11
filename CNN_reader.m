function results = CNN_reader(dataRoot, forceRetrain)
% CNN_reader Train and evaluate a CNN on raw MNIST IDX files.
%
%   RESULTS = CNN_reader() reads files from .\MNIST\raw relative to this
%   function file, trains a simple CNN, tests it, and returns a struct with
%   predicted labels, true labels, confusion matrix, confusion table, and
%   accuracy.
%
%   RESULTS = CNN_reader(dataRoot) uses the provided MNIST raw folder path.
%
%   RESULTS = CNN_reader(dataRoot, forceRetrain) retrains when forceRetrain
%   is true. Otherwise, the function reuses the last saved trained CNN when
%   its training-data root matches dataRoot.

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
	[XTrain, YTrain] = loadMNIST(trainImagesPath, trainLabelsPath, classNames);
	[XTest, YTest]   = loadMNIST(testImagesPath, testLabelsPath, classNames);

	layers = [
		imageInputLayer([28 28 1], 'Name', 'input', 'Normalization', 'none')

		convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
		batchNormalizationLayer('Name', 'bn1')
		reluLayer('Name', 'relu1')
		maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

		convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
		batchNormalizationLayer('Name', 'bn2')
		reluLayer('Name', 'relu2')
		maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

		fullyConnectedLayer(64, 'Name', 'fc1')
		reluLayer('Name', 'relu3')
		fullyConnectedLayer(10, 'Name', 'fc2')
		softmaxLayer('Name', 'softmax')
		classificationLayer('Name', 'output')
	];

	options = trainingOptions('adam', ...
		'InitialLearnRate', 1e-3, ...
		'MaxEpochs', 6, ...
		'MiniBatchSize', 128, ...
		'Shuffle', 'every-epoch', ...
		'ValidationData', {XTest, YTest}, ...
		'ValidationFrequency', 100, ...
		'Verbose', true, ...
		'Plots', 'training-progress');

	cacheFile = getCNNCacheFile();
	[net, loadedFromCache] = tryLoadCNNCache(cacheFile, trainImagesPath, trainLabelsPath, forceRetrain);
	if loadedFromCache
		fprintf('Loaded cached CNN model from: %s\n', getSetFolderPaths(cacheFile));
	else
		fprintf('Training CNN...\n');
		net = trainNetwork(XTrain, YTrain, layers, options);
		saveCNNCache(cacheFile, net, trainImagesPath, trainLabelsPath);
		fprintf('Saved CNN cache to: %s\n', getSetFolderPaths(cacheFile));
	end

	fprintf('Running inference on test set...\n');
	YPred = classify(net, XTest, 'MiniBatchSize', 256);

	accuracy = mean(YPred == YTest) * 100;
	confMat = confusionmat(YTest, YPred);
	confTbl = array2table(confMat, ...
		'VariableNames', cellstr(matlab.lang.makeValidName("Pred_" + classNames)), ...
		'RowNames', cellstr(matlab.lang.makeValidName("True_" + classNames)));

	fprintf('\nTest Accuracy: %.2f%%\n', accuracy);
	disp('Confusion Table (rows=True, cols=Pred):');
	disp(confTbl);

	figure('Name', 'MNIST Confusion Matrix', 'Color', 'w');
	confusionchart(YTest, YPred);
	title(sprintf('MNIST CNN Confusion Matrix (Accuracy: %.2f%%)', accuracy));

	results = struct();
	results.Network = net;
	results.YTrue = YTest;
	results.YPred = YPred;
	results.ConfusionMatrix = confMat;
	results.ConfusionTable = confTbl;
	results.Accuracy = accuracy;
	results.ClassNames = classNames;
end

function cacheFile = getCNNCacheFile()
	here = fileparts(mfilename('fullpath'));
	cacheDir = fullfile(here, 'trained_models');
	if ~isfolder(cacheDir)
		mkdir(cacheDir);
	end
	cacheFile = fullfile(cacheDir, 'cnn_reader_cache.mat');
end

function [net, loadedFromCache] = tryLoadCNNCache(cacheFile, trainImagesPath, trainLabelsPath, forceRetrain)
	net = [];
	loadedFromCache = false;
	if forceRetrain || ~isfile(cacheFile)
		return;
	end

	data = load(cacheFile, 'net', 'cacheMeta');
	if ~isfield(data, 'net') || ~isfield(data, 'cacheMeta')
		return;
	end
	if ~isfield(data.cacheMeta, 'trainImagesPath') || ~isfield(data.cacheMeta, 'trainLabelsPath')
		return;
	end
	if ~strcmp(data.cacheMeta.trainImagesPath, trainImagesPath) || ...
			~strcmp(data.cacheMeta.trainLabelsPath, trainLabelsPath)
		return;
	end

	net = data.net;
	loadedFromCache = true;
end

function saveCNNCache(cacheFile, net, trainImagesPath, trainLabelsPath)
	cacheMeta = struct();
	cacheMeta.trainImagesPath = trainImagesPath;
	cacheMeta.trainLabelsPath = trainLabelsPath;
	cacheMeta.savedAt = char(datetime('now'));
	save(cacheFile, 'net', 'cacheMeta', '-v7.3');
end

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

	images4D = images;
	labelsCat = categorical(labels, 0:9, classNames);
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
