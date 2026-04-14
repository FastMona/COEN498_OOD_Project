function results = MLP_chex(chexRoot, forceRetrain)
% MLP_chex  Train and evaluate a fully-connected MLP on 390x320 CheXpert images.
%
%   RESULTS = MLP_chex() trains on images in .\chex_train relative to this
%   file, then evaluates on a held-out 20% split.
%
%   RESULTS = MLP_chex(chexRoot) uses the provided flat image folder.
%
%   RESULTS = MLP_chex(chexRoot, forceRetrain) forces retraining when true.
%
%   Network:  imageInputLayer([320 390 1])
%             flattenLayer  (-> 124800 features)
%             FC(1024) -> ReLU -> FC(512) -> ReLU -> FC(256) -> ReLU -> FC(128) -> ReLU
%             FC(1) -> regressionLayer
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

    if nargin < 1 || isempty(chexRoot)
        here = fileparts(mfilename('fullpath'));
        chexRoot = fullfile(here, 'chex_train');
    end
    if nargin < 2
        forceRetrain = false;
    end

    if ~isfolder(chexRoot)
        error('MLP_chex:missingFolder', ...
            'Image folder not found: %s\nRun pad_chex.py first.', chexRoot);
    end

    % -----------------------------------------------------------------------
    % Build datastore
    % -----------------------------------------------------------------------
    fprintf('MLP_chex: scanning %s\n', chexRoot);
    imds = imageDatastore(chexRoot, ...
        'FileExtensions', {'.jpg', '.jpeg'}, ...
        'ReadFcn', @readAndPreprocess);
    numImages = numel(imds.Files);
    fprintf('MLP_chex: found %d images\n', numImages);

    if numImages == 0
        error('MLP_chex:noImages', 'No .jpg files found in %s', chexRoot);
    end

    % -----------------------------------------------------------------------
    % Reproducible 80/20 train/test split  (same seed as CNN_chex)
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
    % Architecture  [320 x 390 x 1]  ->  flatten 124800  ->  FC chain -> 1
    % -----------------------------------------------------------------------
    numFeatures = 320 * 390;   % 124 800

    layers = [
        imageInputLayer([320 390 1], 'Name', 'input', 'Normalization', 'none')
        flattenLayer('Name', 'flatten')          % 124800

        fullyConnectedLayer(1024, 'Name', 'fc1')
        reluLayer('Name', 'relu1')

        fullyConnectedLayer(512, 'Name', 'fc2')
        reluLayer('Name', 'relu2')

        fullyConnectedLayer(256, 'Name', 'fc3')
        reluLayer('Name', 'relu3')

        fullyConnectedLayer(128, 'Name', 'fc4')
        reluLayer('Name', 'relu4')

        fullyConnectedLayer(1, 'Name', 'fc5')
        regressionLayer('Name', 'output')
    ];

    fprintf('MLP_chex: architecture  %d -> [1024-512-256-128] -> 1\n', numFeatures);

    options = trainingOptions('adam', ...
        'InitialLearnRate', 1e-4, ...
        'MaxEpochs', 5, ...
        'MiniBatchSize', 64, ...
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
        fprintf('MLP_chex: loaded cached model from %s\n', cacheFile);
    else
        fprintf('MLP_chex: training on %d images...\n', nTrain);
        net = trainNetwork(trainDs, layers, options);
        saveCache(cacheFile, net, chexRoot);
        fprintf('MLP_chex: saved cache to %s\n', cacheFile);
    end

    % -----------------------------------------------------------------------
    % Inference on test split
    % -----------------------------------------------------------------------
    fprintf('MLP_chex: scoring %d test images...\n', nTest);
    scores = predict(net, testImds, 'MiniBatchSize', 128);

    meanScore = mean(scores);
    stdScore  = std(scores);
    fprintf('MLP_chex: test score  mean=%.4f  std=%.4f  (normal target=1.0)\n', ...
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
    cacheFile = fullfile(cacheDir, 'mlp_chex_cache.mat');
end

function arch = archSignature()
% Single source of truth for the architecture string.
% Changing the network layout must be reflected here so the cache is
% automatically invalidated and retraining is triggered.
    arch = '1024-512-256-128';
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
        fprintf('MLP_chex: architecture changed (%s) — discarding cached model.\n', archSignature());
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
        fprintf('MLP_chex: early stop — validation RMSE unchanged at %.6f\n', currRMSE);
        stop = true;
        return;
    end
    prevRMSE = currRMSE;
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
    img = im2single(img);
    if ndims(img) == 2
        img = reshape(img, size(img, 1), size(img, 2), 1);
    end
end
