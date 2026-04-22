clear; clc; close all;

%% USER SETTINGS

dataFolder  = 'C:\Users\David\Documents_local\Repository_local\MATLAB\COEN498_OOD_Project\MNIST_digits';
modelFolder = 'C:\Users\David\Documents_local\Repository_local\MATLAB\COEN498_OOD_Project\trained_models';

trainImageFile = fullfile(dataFolder, 'train-images-idx3-ubyte');
trainLabelFile = fullfile(dataFolder, 'train-labels-idx1-ubyte');
modelFile      = fullfile(modelFolder, 'mnist_cnn_model_with_multilayer_md.mat');

rng('shuffle');

%% CHECK FILES

if ~isfile(trainImageFile)
    error('Training image file not found: %s', trainImageFile);
end

if ~isfile(trainLabelFile)
    error('Training label file not found: %s', trainLabelFile);
end

%% LOAD TRAINING DATA

fprintf('Loading training IDX files...\n');

XTrain = loadMNISTImages(trainImageFile);
YTrain = loadMNISTLabels(trainLabelFile);

if isempty(XTrain) || isempty(YTrain)
    error('Could not load the training dataset.');
end

if size(XTrain,4) ~= numel(YTrain)
    error('Number of training images and labels does not match.');
end

%% PREPROCESS

XTrain = single(XTrain) / 255;
YTrain = categorical(YTrain);

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

%% MULTI-LAYER MAHALANOBIS STATISTICS

featureLayers = {'relu1', 'relu2', 'relu3', 'fc1'};

fprintf('\nExtracting multi-layer training features...\n');
[FTrain, fusionStats] = extractMultiLayerFeatures(net, XTrain, featureLayers, []);

FTrain = double(FTrain);

classes = categories(YTrain);
numClasses = numel(classes);
featureDim = size(FTrain, 2);
N = size(FTrain, 1);

fprintf('Fused feature dimension: %d\n', featureDim);

%% CLASS MEANS

classMeans = zeros(numClasses, featureDim);

for c = 1:numClasses
    idx = (YTrain == classes{c});
    classMeans(c, :) = mean(FTrain(idx, :), 1);
end

%% POOLED COVARIANCE

Sigma = zeros(featureDim, featureDim);

for c = 1:numClasses
    idx = find(YTrain == classes{c});
    Fc = FTrain(idx, :);
    Mc = classMeans(c, :);
    Dc = Fc - Mc;
    Sigma = Sigma + (Dc' * Dc);
end

Sigma = Sigma / N;

%% REGULARIZATION

lambda = 1e-2;   
SigmaReg = Sigma + lambda * eye(featureDim);

invSigma = pinv(SigmaReg);

%% COMPUTE TRAINING MD SCORES

fprintf('Computing training Mahalanobis scores...\n');
trainMDScores = zeros(N,1);

for i = 1:N
    z = FTrain(i, :);
    dists = zeros(numClasses,1);

    for c = 1:numClasses
        diffVec = z - classMeans(c, :);
        dists(c) = diffVec * invSigma * diffVec';
    end

    trainMDScores(i) = min(dists);
end

thresholdPercentile = 97.5;
mdThreshold = prctile(trainMDScores, thresholdPercentile);

fprintf('MD threshold (%.1f percentile of ID train scores): %.4f\n', ...
    thresholdPercentile, mdThreshold);

%% SAVE MODEL + MD STATS

mdStats.featureLayers = featureLayers;
mdStats.fusionStats = fusionStats;       
mdStats.classMeans = classMeans;
mdStats.invSigma = invSigma;
mdStats.classes = classes;
mdStats.lambda = lambda;
mdStats.trainMDScores = trainMDScores;
mdStats.mdThreshold = mdThreshold;
mdStats.thresholdPercentile = thresholdPercentile;

save(modelFile, 'net', 'mdStats', '-v7.3');

fprintf('\nModel + multi-layer MD stats saved to:\n%s\n', modelFile);

%% LOCAL FUNCTIONS

function [F, fusionStats] = extractMultiLayerFeatures(net, X, layerNames, fusionStatsIn)

    numLayers = numel(layerNames);
    featureBlocks = cell(numLayers,1);

    if isempty(fusionStatsIn)
        fusionStats = struct([]);
    else
        fusionStats = fusionStatsIn;
    end

    for k = 1:numLayers
        layerName = layerNames{k};
        A = activations(net, X, layerName);

        % Case 1: convolutional feature maps: H x W x C x N
        if ndims(A) == 4
            % global average pooling over H and W
            A = squeeze(mean(mean(A,1),2));  % C x N or C x 1 if N=1

            if isvector(A)
                A = A(:)';  % 1 x C
            else
                A = A';     % N x C
            end

        % Case 2: vector / FC output
        else
            A = activations(net, X, layerName, 'OutputAs', 'rows'); % N x D
        end

        A = double(A);

        % Normalize each layer block independently before concatenation
        if isempty(fusionStatsIn)
            mu = mean(A, 1);
            sigma = std(A, 0, 1);
            sigma(sigma < 1e-8) = 1;

            fusionStats(k).layerName = layerName;
            fusionStats(k).mu = mu;
            fusionStats(k).sigma = sigma;
        else
            mu = fusionStats(k).mu;
            sigma = fusionStats(k).sigma;
        end

        A = (A - mu) ./ sigma;
        featureBlocks{k} = A;
    end

    F = cat(2, featureBlocks{:});
end

function images = loadMNISTImages(filename)
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Cannot open image file: %s', filename);
    end

    magicNum = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magicNum ~= 2051
        fclose(fid);
        error('Invalid MNIST image file: %s', filename);
    end

    numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
    numRows   = fread(fid, 1, 'int32', 0, 'ieee-be');
    numCols   = fread(fid, 1, 'int32', 0, 'ieee-be');

    rawData = fread(fid, inf, 'uint8=>uint8');
    fclose(fid);

    expectedNumPixels = numImages * numRows * numCols;
    if numel(rawData) ~= expectedNumPixels
        error('Image file size does not match header information.');
    end

    images = reshape(rawData, numCols, numRows, 1, numImages);
    images = permute(images, [2 1 3 4]);
end

function labels = loadMNISTLabels(filename)
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Cannot open label file: %s', filename);
    end

    magicNum = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magicNum ~= 2049
        fclose(fid);
        error('Invalid MNIST label file: %s', filename);
    end

    numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
    labels = fread(fid, inf, 'uint8=>uint8');
    fclose(fid);

    if numel(labels) ~= numLabels
        error('Label file size does not match header information.');
    end
end