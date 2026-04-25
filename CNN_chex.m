function results = CNN_chex(chexRoot, forceRetrain)
% CNN_chex  Train and evaluate a CNN on 390x320 CheXpert chest X-ray images.
%
%   RESULTS = CNN_chex() trains on images in .\chex_train relative to this
%   file, then evaluates on a held-out 20% split.
%
%   RESULTS = CNN_chex(chexRoot) uses the provided flat image folder.
%
%   RESULTS = CNN_chex(chexRoot, forceRetrain) forces retraining when true.
%
%   Network:  imageInputLayer([320 390 1])
%             4x (Conv3x3 -> BN -> ReLU -> MaxPool2x2)
%             FC(256) -> ReLU -> FC(1) -> regressionLayer
%
%   All training labels are 1.0 (normal).  At inference the raw output is
%   the anomaly score: values near 1.0 indicate normal; lower values flag
%   potential anomalies.
%
%   Output struct fields:
%     .Network    – trained SeriesNetwork
%     .TestFiles  – cell array of test image file paths
%     .Scores     – Nx1 raw regression outputs for test images
%     .MeanScore  – mean score over test set
%     .StdScore   – std of scores over test set
%     .ChexRoot   – resolved image folder path

    chexRoot = resolveChexRoot(chexRoot);
    if nargin < 2
        forceRetrain = false;
    end

    if ~isfolder(chexRoot)
        error('CNN_chex:missingFolder', ...
            'Image folder not found: %s\nUse an absolute path or a path relative to this file.', chexRoot);
    end

    % -----------------------------------------------------------------------
    % Build datastore
    % -----------------------------------------------------------------------
    fprintf('CNN_chex: scanning %s\n', chexRoot);
    imds = imageDatastore(chexRoot, ...
        'FileExtensions', {'.jpg', '.jpeg'}, ...
        'ReadFcn', @readAndPreprocess);
    numImages = numel(imds.Files);
    fprintf('CNN_chex: found %d images\n', numImages);

    if numImages == 0
        error('CNN_chex:noImages', 'No .jpg files found in %s', chexRoot);
    end

    % -----------------------------------------------------------------------
    % Reproducible 80/20 train/test split
    % -----------------------------------------------------------------------
    rng(42);
    perm   = randperm(numImages);
    nTrain = floor(0.8 * numImages);
    nTest  = numImages - nTrain;

    trainImds = imageDatastore(imds.Files(perm(1:nTrain)), ...
        'ReadFcn', @readAndPreprocess);
    testImds  = imageDatastore(imds.Files(perm(nTrain+1:end)), ...
        'ReadFcn', @readAndPreprocess);

    % Regression targets: 1.0 = normal
    trainTargets = ones(nTrain, 1);
    testTargets  = ones(nTest,  1);

    trainDs = combine(trainImds, arrayDatastore(trainTargets));
    testDs  = combine(testImds,  arrayDatastore(testTargets));

    % -----------------------------------------------------------------------
    % Architecture  [320 x 390 x 1]
    %   pool1 -> [160 x 195 x 16]
    %   pool2 -> [80  x  97 x 32]
    %   pool3 -> [40  x  48 x 64]
    %   pool4 -> [20  x  24 x 128]
    %   flatten -> 61440 -> FC(256) -> FC(1)
    % -----------------------------------------------------------------------
    layers = [
        imageInputLayer([320 390 1], 'Name', 'input', 'Normalization', 'none')

        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4')
        batchNormalizationLayer('Name', 'bn4')
        reluLayer('Name', 'relu4')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')

        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu5')

        fullyConnectedLayer(1, 'Name', 'fc2')
        regressionLayer('Name', 'output')
    ];

    options = trainingOptions('adam', ...
        'InitialLearnRate', 1e-4, ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', testDs, ...
        'ValidationFrequency', 50, ...
        'OutputFcn', @earlyStopOnRMSERepeat, ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    % -----------------------------------------------------------------------
    % Train or load from cache
    % -----------------------------------------------------------------------
    cacheFile = getCacheFile();
    [net, loadedFromCache] = tryLoadCache(cacheFile, chexRoot, forceRetrain);
    if loadedFromCache
        fprintf('CNN_chex: loaded cached model from %s\n', cacheFile);
    else
        fprintf('CNN_chex: training on %d images...\n', nTrain);
        net = trainNetwork(trainDs, layers, options);
        saveCache(cacheFile, net, chexRoot);
        fprintf('CNN_chex: saved cache to %s\n', cacheFile);
    end

    % -----------------------------------------------------------------------
    % Inference on test split
    % -----------------------------------------------------------------------
    fprintf('CNN_chex: scoring %d test images...\n', nTest);
    scores = predict(net, testImds, 'MiniBatchSize', 64);

    meanScore = mean(scores);
    stdScore  = std(scores);
    fprintf('CNN_chex: test score  mean=%.4f  std=%.4f  (normal target=1.0)\n', ...
        meanScore, stdScore);

    results = struct();
    results.Network   = net;
    results.TestFiles = imds.Files(perm(nTrain+1:end));
    results.Scores    = scores;
    results.MeanScore = meanScore;
    results.StdScore  = stdScore;
    results.ChexRoot  = chexRoot;
end

% ==========================================================================
% Cache helpers
% ==========================================================================

function cacheFile = getCacheFile()
    here     = fileparts(mfilename('fullpath'));
    cacheDir = fullfile(here, 'trained_models');
    if ~isfolder(cacheDir)
        mkdir(cacheDir);
    end
    cacheFile = fullfile(cacheDir, 'cnn_chex_cache.mat');
end

function arch = archSignature()
% Single source of truth for the CNN architecture string.
% Update this string whenever the layer layout changes so the cache is
% automatically invalidated and retraining is triggered.
    arch = 'conv16-conv32-conv64-conv128-fc256';
end

function [net, loadedFromCache] = tryLoadCache(cacheFile, chexRoot, forceRetrain)
    net = [];
    loadedFromCache = false;
    if forceRetrain || ~isfile(cacheFile)
        return;
    end
    data = load(cacheFile, 'net', 'cacheMeta');
    if ~isfield(data, 'net') || ~isfield(data, 'cacheMeta')
        return;
    end
    if ~isfield(data.cacheMeta, 'chexRoot') || ~strcmp(data.cacheMeta.chexRoot, chexRoot)
        return;
    end
    if ~isfield(data.cacheMeta, 'arch') || ~strcmp(data.cacheMeta.arch, archSignature())
        fprintf('CNN_chex: architecture changed (%s) — discarding cached model.\n', archSignature());
        return;
    end
    net = data.net;
    loadedFromCache = true;
end

function saveCache(cacheFile, net, chexRoot)
    cacheMeta.chexRoot = chexRoot;
    cacheMeta.arch     = archSignature();
    cacheMeta.savedAt  = char(datetime('now'));
    save(cacheFile, 'net', 'cacheMeta', '-v7.3');
end

% ==========================================================================
% Early-stopping output function
% ==========================================================================

function stop = earlyStopOnRMSERepeat(info)
% Stop training when validation RMSE does not change between validation checks.
    persistent prevRMSE
    stop = false;
    if strcmp(info.State, 'start')
        prevRMSE = NaN;
        return;
    end
    currRMSE = info.ValidationRMSE;
    if ~isscalar(currRMSE) || isnan(currRMSE)
        return;
    end
    if ~isnan(prevRMSE) && abs(currRMSE - prevRMSE) < 1e-6
        fprintf('CNN_chex: early stop — validation RMSE unchanged at %.6f\n', currRMSE);
        stop = true;
        return;
    end
    prevRMSE = info.ValidationRMSE;
end

% ==========================================================================
% Image pre-processing (used as imageDatastore ReadFcn)
% ==========================================================================

function img = readAndPreprocess(filename)
% Read a CheXpert JPEG, ensure single-channel [320x390x1] single in [0,1].
    img = imread(filename);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2single(img);        % uint8 -> single [0,1]
    if ndims(img) == 2
        img = reshape(img, size(img, 1), size(img, 2), 1);
    end
end

function chexRoot = resolveChexRoot(chexRoot)
    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(chexRoot)
        chexRoot = fullfile(here, 'chex_train');
        return;
    end

    chexRoot = char(string(chexRoot));
    if ~isfolder(chexRoot)
        candidate = fullfile(here, chexRoot);
        if isfolder(candidate)
            chexRoot = candidate;
        end
    end
end
