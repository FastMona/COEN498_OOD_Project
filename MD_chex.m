function results = MD_chex(testInput, chexRoot, vigilance, forceRetrain)
% MD_chex  Three-stage OOD detector for CheXpert chest X-rays.
%
%   Stage 1 – Pixel-space MD prefilter (Is this a chest X-ray at all?)
%     Builds a PCA manifold from raw training pixels and rejects inputs
%     whose Mahalanobis confidence falls below vigilance.
%
%   Stage 2 – NN regression scoring (CNN_chex + MLP_chex)
%     Accepted images are scored by both networks.  Normal target = 1.0.
%
%   Stage 3 – Latent-space MD filter (Does this X-ray look normal?)
%     Extracts penultimate-layer activations from each NN (relu5 for CNN,
%     relu3 for MLP), builds a PCA manifold over those features, then tests
%     whether the accepted images sit within the normal latent distribution.
%
%   RESULTS = MD_chex(testInput)
%     testInput : path to a flat folder of 390x320 .jpg files
%                 OR an imageDatastore of pre-loaded images
%
%   RESULTS = MD_chex(testInput, chexRoot)
%     chexRoot  : path to the 390x320 training image folder (chex_train).
%                 Defaults to .\chex_train relative to this file.
%
%   RESULTS = MD_chex(testInput, chexRoot, vigilance)
%     vigilance : acceptance confidence threshold in [0,1]. Default 0.5.
%                 Applied at both Stage 1 and Stage 3.
%
%   RESULTS = MD_chex(testInput, chexRoot, vigilance, forceRetrain)
%     forceRetrain : pass true to ignore all caches and retrain from scratch.
%
%   Output struct fields:
%     .Vigilance           – threshold used at both stages
%     .NumSamples          – total test images
%     .TestFiles           – file paths (if testInput was a folder)
%
%     Stage 1 outputs:
%     .PixelModel          – pixel-space PCA manifold struct
%     .MDScores            – Nx1 Stage-1 confidence (all samples)
%     .NormalizedDist      – Nx1 Stage-1 normalised Mahalanobis distance
%     .Accepted            – Nx1 logical: passed Stage 1
%     .IsOOD               – Nx1 logical: rejected at Stage 1
%     .AcceptedCount       – number passed Stage 1
%     .RejectedCount       – number rejected at Stage 1
%
%     Stage 2 outputs (NaN for Stage-1 rejects):
%     .CNN                 – CNN_chex results struct
%     .MLP                 – MLP_chex results struct
%     .CNNScores           – Nx1 CNN regression scores
%     .MLPScores           – Nx1 MLP regression scores
%     .CombinedScores      – Nx1 mean of CNN and MLP scores
%
%     Stage 3 outputs (NaN for Stage-1 rejects):
%     .LatentModel         – struct with .CNN and .MLP latent manifolds
%     .CNNLatentScores     – Nx1 Stage-3 confidence from CNN latent space
%     .MLPLatentScores     – Nx1 Stage-3 confidence from MLP latent space
%     .LatentScores        – Nx1 mean latent confidence
%     .LatentAccepted      – Nx1 logical: passed BOTH Stage 1 and Stage 3
%     .LatentAcceptedCount – number passing all three stages
%     .LatentRejectedCount – number rejected at Stage 3 (were normal X-rays
%                            but have abnormal latent features)

    if nargin < 1 || isempty(testInput)
        error('MD_chex:noInput', 'Provide a test folder path or imageDatastore.');
    end
    if nargin < 2 || isempty(chexRoot)
        here = fileparts(mfilename('fullpath'));
        chexRoot = fullfile(here, 'chex_train');
    end
    if nargin < 3 || isempty(vigilance)
        vigilance = 0.5;
    end
    if nargin < 4
        forceRetrain = false;
    end
    if vigilance < 0 || vigilance > 1
        error('MD_chex:badVigilance', 'vigilance must be between 0 and 1.');
    end

    % -----------------------------------------------------------------------
    % Stage 1 setup: pixel-space PCA manifold
    % -----------------------------------------------------------------------
    pixelModel = loadOrTrainPixelManifold(chexRoot, forceRetrain);

    % -----------------------------------------------------------------------
    % Stage 2 setup: load / train NN models
    % -----------------------------------------------------------------------
    fprintf('MD_chex: loading CNN model...\n');
    cnnResults = CNN_chex(chexRoot, forceRetrain);

    fprintf('MD_chex: loading MLP model...\n');
    mlpResults = MLP_chex(chexRoot, forceRetrain);

    cnnNet = cnnResults.Network;
    mlpNet = mlpResults.Network;

    % -----------------------------------------------------------------------
    % Stage 3 setup: latent-space PCA manifolds
    % -----------------------------------------------------------------------
    latentModel = loadOrTrainLatentManifold(chexRoot, cnnNet, mlpNet, forceRetrain);

    % -----------------------------------------------------------------------
    % Prepare test datastore
    % -----------------------------------------------------------------------
    [testImds, testFiles] = prepareTestInput(testInput);
    numSamples = numel(testImds.Files);
    fprintf('MD_chex: scoring %d test images...\n', numSamples);

    % -----------------------------------------------------------------------
    % Stage 1: pixel-space MD — reject non-X-ray inputs
    % -----------------------------------------------------------------------
    XTest = extractPixelFeatures(testImds);
    [mdScores, normalizedDist] = scoreManifold(pixelModel, XTest);

    accepted      = mdScores >= vigilance;
    acceptedCount = sum(accepted);
    rejectedCount = numSamples - acceptedCount;

    fprintf('\n--- Stage 1: pixel-space prefilter ---\n');
    fprintf('  Vigilance   : %.2f\n', vigilance);
    fprintf('  Test images : %d\n',  numSamples);
    fprintf('  Accepted    : %d (%.1f%%)\n', acceptedCount, 100 * acceptedCount / numSamples);
    fprintf('  Rejected    : %d (%.1f%%)\n', rejectedCount, 100 * rejectedCount / numSamples);

    % -----------------------------------------------------------------------
    % Stage 2: NN regression on accepted samples
    % -----------------------------------------------------------------------
    cnnScores      = nan(numSamples, 1, 'single');
    mlpScores      = nan(numSamples, 1, 'single');
    combinedScores = nan(numSamples, 1, 'single');

    cnnLatentScores = nan(numSamples, 1, 'single');
    mlpLatentScores = nan(numSamples, 1, 'single');
    latentScores    = nan(numSamples, 1, 'single');
    latentAccepted  = false(numSamples, 1);

    if acceptedCount > 0
        acceptedFiles = testFiles(accepted);
        acceptedImds  = imageDatastore(acceptedFiles, 'ReadFcn', @readAndPreprocess);

        fprintf('\n--- Stage 2: NN regression ---\n');
        cnnAccepted  = predict(cnnNet, acceptedImds, 'MiniBatchSize', 32);
        mlpAccepted  = predict(mlpNet, acceptedImds, 'MiniBatchSize', 64);
        combAccepted = (cnnAccepted + mlpAccepted) / 2;

        cnnScores(accepted)      = cnnAccepted;
        mlpScores(accepted)      = mlpAccepted;
        combinedScores(accepted) = combAccepted;

        fprintf('  CNN  score: mean=%.4f  std=%.4f\n', mean(cnnAccepted),  std(cnnAccepted));
        fprintf('  MLP  score: mean=%.4f  std=%.4f\n', mean(mlpAccepted),  std(mlpAccepted));
        fprintf('  Combined  : mean=%.4f  std=%.4f\n', mean(combAccepted), std(combAccepted));
        fprintf('  (normal target = 1.0; lower = more abnormal)\n');

        % -------------------------------------------------------------------
        % Stage 3: latent-space MD — flag abnormal X-rays
        % -------------------------------------------------------------------
        fprintf('\n--- Stage 3: latent-space filter ---\n');
        reset(acceptedImds);
        cnnLatent = extractLatentFeatures(cnnNet, acceptedImds, 'relu5');
        reset(acceptedImds);
        mlpLatent = extractLatentFeatures(mlpNet, acceptedImds, 'relu3');

        [cnnLatConf, ~] = scoreManifold(latentModel.CNN, cnnLatent);
        [mlpLatConf, ~] = scoreManifold(latentModel.MLP, mlpLatent);
        combLatConf     = (cnnLatConf + mlpLatConf) / 2;

        cnnLatentScores(accepted) = cnnLatConf;
        mlpLatentScores(accepted) = mlpLatConf;
        latentScores(accepted)    = combLatConf;

        latentPassMask           = combLatConf >= vigilance;
        latentAccepted(accepted) = latentPassMask;

        latAcceptedCount = sum(latentPassMask);
        latRejectedCount = acceptedCount - latAcceptedCount;

        fprintf('  Vigilance   : %.2f\n', vigilance);
        fprintf('  Accepted    : %d (%.1f%% of Stage-1 accepted)\n', ...
            latAcceptedCount, 100 * latAcceptedCount / acceptedCount);
        fprintf('  Rejected    : %d (%.1f%% of Stage-1 accepted)\n', ...
            latRejectedCount, 100 * latRejectedCount / acceptedCount);
        fprintf('  CNN  latent : mean=%.4f  std=%.4f\n', mean(cnnLatConf), std(cnnLatConf));
        fprintf('  MLP  latent : mean=%.4f  std=%.4f\n', mean(mlpLatConf), std(mlpLatConf));
    else
        fprintf('  All samples rejected at Stage 1 — Stages 2 and 3 not run.\n');
        latAcceptedCount = 0;
        latRejectedCount = 0;
    end

    % -----------------------------------------------------------------------
    % Pack results
    % -----------------------------------------------------------------------
    results = struct();
    results.Vigilance           = vigilance;
    results.NumSamples          = numSamples;
    results.TestFiles           = testFiles;

    results.PixelModel          = pixelModel;
    results.MDScores            = mdScores;
    results.NormalizedDist      = normalizedDist;
    results.Accepted            = accepted;
    results.IsOOD               = ~accepted;
    results.AcceptedCount       = acceptedCount;
    results.RejectedCount       = rejectedCount;

    results.CNN                 = cnnResults;
    results.MLP                 = mlpResults;
    results.CNNScores           = cnnScores;
    results.MLPScores           = mlpScores;
    results.CombinedScores      = combinedScores;

    results.LatentModel         = latentModel;
    results.CNNLatentScores     = cnnLatentScores;
    results.MLPLatentScores     = mlpLatentScores;
    results.LatentScores        = latentScores;
    results.LatentAccepted      = latentAccepted;
    results.LatentAcceptedCount = latAcceptedCount;
    results.LatentRejectedCount = latRejectedCount;
end

% ==========================================================================
% Stage 1: pixel-space manifold
% ==========================================================================

function model = loadOrTrainPixelManifold(chexRoot, forceRetrain)
    cacheFile = getPixelCacheFile();
    if ~forceRetrain && isfile(cacheFile)
        data = load(cacheFile, 'model', 'cacheMeta');
        if isfield(data, 'model') && isfield(data, 'cacheMeta') && ...
                isfield(data.cacheMeta, 'chexRoot') && strcmp(data.cacheMeta.chexRoot, chexRoot)
            model = data.model;
            fprintf('MD_chex: loaded cached pixel manifold from %s\n', cacheFile);
            return;
        end
    end

    fprintf('MD_chex: building pixel-space PCA manifold from %s\n', chexRoot);
    imds = imageDatastore(chexRoot, ...
        'FileExtensions', {'.jpg', '.jpeg'}, ...
        'ReadFcn', @readAndPreprocess);
    if numel(imds.Files) == 0
        error('MD_chex:noTrainImages', 'No .jpg files found in chexRoot: %s', chexRoot);
    end

    fprintf('MD_chex: extracting pixel features from %d training images...\n', numel(imds.Files));
    XTrain = extractPixelFeatures(imds);
    model  = trainManifold(XTrain);

    cacheMeta.chexRoot = chexRoot;
    cacheMeta.savedAt  = char(datetime('now'));
    save(cacheFile, 'model', 'cacheMeta', '-v7.3');
    fprintf('MD_chex: saved pixel manifold to %s\n', cacheFile);
end

function cacheFile = getPixelCacheFile()
    here     = fileparts(mfilename('fullpath'));
    cacheDir = fullfile(here, 'trained_models');
    if ~isfolder(cacheDir)
        mkdir(cacheDir);
    end
    cacheFile = fullfile(cacheDir, 'md_chex_cache.mat');
end

% ==========================================================================
% Stage 3: latent-space manifolds
% ==========================================================================

function latentModel = loadOrTrainLatentManifold(chexRoot, cnnNet, mlpNet, forceRetrain)
    cacheFile = getLatentCacheFile();
    if ~forceRetrain && isfile(cacheFile)
        data = load(cacheFile, 'latentModel', 'cacheMeta');
        if isfield(data, 'latentModel') && isfield(data, 'cacheMeta') && ...
                isfield(data.cacheMeta, 'chexRoot') && strcmp(data.cacheMeta.chexRoot, chexRoot)
            latentModel = data.latentModel;
            fprintf('MD_chex: loaded cached latent manifold from %s\n', cacheFile);
            return;
        end
    end

    fprintf('MD_chex: building latent-space manifolds from %s\n', chexRoot);
    imds = imageDatastore(chexRoot, ...
        'FileExtensions', {'.jpg', '.jpeg'}, ...
        'ReadFcn', @readAndPreprocess);

    fprintf('MD_chex: extracting CNN latent features (relu5, 256-d) from %d images...\n', numel(imds.Files));
    cnnFeatures = extractLatentFeatures(cnnNet, imds, 'relu5');

    reset(imds);
    fprintf('MD_chex: extracting MLP latent features (relu3, 128-d) from %d images...\n', numel(imds.Files));
    mlpFeatures = extractLatentFeatures(mlpNet, imds, 'relu3');

    latentModel.CNN = trainManifold(cnnFeatures);
    latentModel.MLP = trainManifold(mlpFeatures);

    cacheMeta.chexRoot = chexRoot;
    cacheMeta.savedAt  = char(datetime('now'));
    save(cacheFile, 'latentModel', 'cacheMeta', '-v7.3');
    fprintf('MD_chex: saved latent manifold to %s\n', cacheFile);
end

function cacheFile = getLatentCacheFile()
    here     = fileparts(mfilename('fullpath'));
    cacheDir = fullfile(here, 'trained_models');
    if ~isfolder(cacheDir)
        mkdir(cacheDir);
    end
    cacheFile = fullfile(cacheDir, 'md_chex_latent_cache.mat');
end

function features = extractLatentFeatures(net, imds, layerName)
% Extract penultimate-layer activations from net over all images in imds.
% Returns N x D single matrix (D = 256 for relu5/CNN, 128 for relu3/MLP).
    reset(imds);
    features = activations(net, imds, layerName, ...
        'MiniBatchSize', 32, 'OutputAs', 'rows');
    features = single(features);
end

% ==========================================================================
% Shared: manifold training and scoring
% ==========================================================================

function model = trainManifold(XTrain)
% Build a single-class PCA manifold.  Works for both pixel features
% (N x 124800) and latent features (N x 256 or N x 128).
% Uses svds to compute only the top NUM_COMPONENTS singular vectors.
    NUM_COMPONENTS = 50;

    mu        = mean(XTrain, 1);
    XCentered = XTrain - mu;

    numComp = min(NUM_COMPONENTS, min(size(XCentered)) - 1);
    [~, S, V] = svds(XCentered, numComp);
    basis      = V;

    singularValues = diag(S);
    latent = (singularValues .^ 2) / max(size(XTrain, 1) - 1, 1);
    latent = max(latent, 1e-6);

    proj = XCentered * basis;
    % ||residual||^2 = ||centered||^2 - ||proj||^2  (basis is orthonormal)
    residualEnergy = (sum(XCentered .^ 2, 2) - sum(proj .^ 2, 2)) / size(XTrain, 2);
    residualVar    = max(median(residualEnergy), 1e-6);

    trainDist     = mahalanobisToManifold(XTrain, mu, basis, latent, residualVar);
    distanceScale = max(prctile(trainDist, 95), 1e-6);

    model.NumComponents = numComp;
    model.mu            = mu;
    model.basis         = basis;
    model.latent        = latent;
    model.residualVar   = residualVar;
    model.distanceScale = distanceScale;
end

function [confidence, normalizedDist] = scoreManifold(model, XTest)
    dist           = mahalanobisToManifold(XTest, model.mu, model.basis, ...
                                           model.latent, model.residualVar);
    normalizedDist = dist ./ max(model.distanceScale, 1e-6);
    confidence     = exp(-0.5 * normalizedDist .^ 2);
end

function distance = mahalanobisToManifold(X, mu, basis, latent, residualVar)
    centered = X - mu;
    proj     = centered * basis;

    latent         = reshape(max(latent, 1e-6), 1, []);
    subspaceTerm   = sum((proj .^ 2) ./ latent, 2) / numel(latent);
    % Orthogonality of basis avoids forming recon and residual explicitly
    residualEnergy = (sum(centered .^ 2, 2) - sum(proj .^ 2, 2)) / size(X, 2);
    residualTerm   = residualEnergy / max(residualVar, 1e-6);
    distance       = sqrt(max(subspaceTerm + residualTerm, 0));
end

% ==========================================================================
% Feature extraction
% ==========================================================================

function X = extractPixelFeatures(imds)
% Flatten all images to N x 124800 single matrix at full resolution.
    reset(imds);
    numImages = numel(imds.Files);
    numPixels = 320 * 390;
    X = zeros(numImages, numPixels, 'single');
    for i = 1:numImages
        img     = read(imds);       % [320 390 1] single
        X(i, :) = img(:)';
    end
end

% ==========================================================================
% Helpers
% ==========================================================================

function [imds, files] = prepareTestInput(testInput)
    if ischar(testInput) || (isstring(testInput) && isscalar(testInput))
        folderPath = char(string(testInput));
        if ~isfolder(folderPath)
            error('MD_chex:missingFolder', 'Test folder not found: %s', folderPath);
        end
        imds  = imageDatastore(folderPath, ...
            'FileExtensions', {'.jpg', '.jpeg'}, ...
            'ReadFcn', @readAndPreprocess);
        files = imds.Files;
    elseif isa(testInput, 'matlab.io.datastore.ImageDatastore')
        imds  = testInput;
        files = imds.Files;
    else
        error('MD_chex:unsupportedInput', ...
            'testInput must be a folder path string or an imageDatastore.');
    end
    if numel(imds.Files) == 0
        error('MD_chex:noImages', 'No .jpg files found in the test input.');
    end
end

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
