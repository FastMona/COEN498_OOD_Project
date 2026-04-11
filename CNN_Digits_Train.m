clear; clc; close all;

%% USER SETTINGS

dataFolder = getSetFolderPaths('resolve', 'trainRoot', '');

modelFile = fullfile(dataFolder, 'mnist_cnn_model.mat');

rng('shuffle');

%% DETECT FORMAT AND LOAD

idxImagesFile = fullfile(dataFolder, 'train-images-idx3-ubyte');
idxLabelsFile = fullfile(dataFolder, 'train-labels-idx1-ubyte');
csvFile       = fullfile(dataFolder, 'mnist_train.csv');

if isfile(idxImagesFile) && isfile(idxLabelsFile)
    fprintf('Detected IDX format. Loading from: %s\n', dataFolder);
    [XTrain, YTrain] = loadIDX(idxImagesFile, idxLabelsFile);
elseif isfile(csvFile)
    fprintf('Detected CSV format. Loading from: %s\n', csvFile);
    [XTrain, YTrain] = loadCSV(csvFile);
else
    error(['No recognised training data found in:\n  %s\n' ...
           'Expected ''train-images-idx3-ubyte'' + ''train-labels-idx1-ubyte'' (IDX) ' ...
           'or ''mnist_train.csv'' (CSV).'], dataFolder);
end

fprintf('Training samples: %d\n', numel(YTrain));

%% DEFINE CNN

layers = [
    imageInputLayer([28 28 1], 'Normalization', 'none', 'Name', 'input')

    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(10, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% TRAIN OPTIONS

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% TRAIN NETWORK

fprintf('Training CNN...\n');
net = trainNetwork(XTrain, YTrain, layers, options);

%% SAVE MODEL

save(modelFile, 'net');

fprintf('\nModel saved to:\n%s\n', modelFile);

%% LOCAL FUNCTIONS

function [XTrain, YTrain] = loadCSV(csvFile)
    trainData = readmatrix(csvFile);
    if isempty(trainData)
        error('Could not read CSV file: %s', csvFile);
    end
    if size(trainData, 2) ~= 785
        error('Expected 785 columns (1 label + 784 pixels) in %s, got %d.', csvFile, size(trainData,2));
    end
    labels = trainData(:, 1);
    pixels = single(trainData(:, 2:end)) / 255;
    XTrain = reshape(pixels', 28, 28, 1, []);
    YTrain = categorical(labels, 0:9, string(0:9));
end

function [XTrain, YTrain] = loadIDX(imagesFile, labelsFile)
    XTrain = readIDXImages(imagesFile);
    labels = readIDXLabels(labelsFile);
    YTrain = categorical(labels, 0:9, string(0:9));
end

function images4D = readIDXImages(filePath)
    fid = fopen(filePath, 'rb');
    if fid < 0
        error('Could not open image file: %s', filePath);
    end
    cleaner = onCleanup(@() fclose(fid));

    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magic ~= 2051
        error('Invalid IDX image magic number in %s (got %d).', filePath, magic);
    end

    numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
    numRows   = fread(fid, 1, 'int32', 0, 'ieee-be');
    numCols   = fread(fid, 1, 'int32', 0, 'ieee-be');

    pixels = fread(fid, numImages * numRows * numCols, 'uint8=>single');
    if numel(pixels) ~= numImages * numRows * numCols
        error('Unexpected end of image file: %s', filePath);
    end

    images3D = permute(reshape(pixels, [numCols, numRows, numImages]), [2, 1, 3]);
    images4D = reshape(images3D ./ 255, [numRows, numCols, 1, numImages]);
end

function labels = readIDXLabels(filePath)
    fid = fopen(filePath, 'rb');
    if fid < 0
        error('Could not open label file: %s', filePath);
    end
    cleaner = onCleanup(@() fclose(fid));

    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magic ~= 2049
        error('Invalid IDX label magic number in %s (got %d).', filePath, magic);
    end

    numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
    labels = fread(fid, numLabels, 'uint8=>double');
    if numel(labels) ~= numLabels
        error('Unexpected end of label file: %s', filePath);
    end
end
