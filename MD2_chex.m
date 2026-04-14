function results = MD2_chex(acceptedFiles, cnnNet, mlpNet, chexRoot, vigilance3, forceRetrain)
% MD2_chex  Stage 3: latent-space Mahalanobis Distance filter.
%
%   Extracts penultimate-layer activations from CNN and MLP networks for
%   each image in acceptedFiles, builds (or loads from cache) PCA manifolds
%   over those features, and tests whether each image sits within the normal
%   latent distribution.
%
%   Both networks must independently clear vigilance3 (AND logic).
%   Averaging before thresholding would let a strong CNN mask a weak MLP
%   result, which is not appropriate for one-class detection.
%
%   RESULTS = MD2_chex(acceptedFiles, cnnNet, mlpNet, chexRoot, vigilance3, forceRetrain)
%     acceptedFiles : M-element cell array of image paths to score
%                     (typically the Stage-1-accepted subset)
%     cnnNet        : trained CNN_chex SeriesNetwork  (relu5 = 256-d features)
%     mlpNet        : trained MLP_chex SeriesNetwork  (relu4 = 128-d features)
%     chexRoot      : flat folder of 390x320 training .jpg files
%     vigilance3    : acceptance confidence in [0,1]  (default 0.7)
%     forceRetrain  : true forces manifold rebuild     (default false)
%
%   Output struct fields (all M x 1, one entry per accepted file):
%     .NumSamples           – M (number of files scored)
%     .Vigilance3           – threshold used
%     .LatentModel          – struct with .CNN and .MLP manifolds
%     .CNNLatentScores      – Mx1 latent confidence from CNN (relu5)
%     .MLPLatentScores      – Mx1 latent confidence from MLP (relu4)
%     .LatentScores         – Mx1 mean of CNN and MLP latent confidence
%     .LatentAccepted       – Mx1 logical: passed Stage 3 (both nets)
%     .LatentAcceptedCount  – number accepted
%     .LatentRejectedCount  – number rejected

    if nargin < 1 || isempty(acceptedFiles)
        error('MD2_chex:noInput', 'Provide a cell array of accepted file paths.');
    end
    if nargin < 4 || isempty(chexRoot)
        here     = fileparts(mfilename('fullpath'));
        chexRoot = fullfile(here, 'chex_train');
    end
    if nargin < 5 || isempty(vigilance3)
        vigilance3 = 0.7;
    end
    if nargin < 6
        forceRetrain = false;
    end
    if vigilance3 < 0 || vigilance3 > 1
        error('MD2_chex:badVigilance3', 'vigilance3 must be in [0, 1].');
    end

    % -----------------------------------------------------------------------
    % Load or build latent-space PCA manifolds
    % -----------------------------------------------------------------------
    latentModel = loadOrTrainLatentManifold(chexRoot, cnnNet, mlpNet, forceRetrain);

    % -----------------------------------------------------------------------
    % Score accepted images
    % -----------------------------------------------------------------------
    M    = numel(acceptedFiles);
    imds = imageDatastore(acceptedFiles, 'ReadFcn', @readAndPreprocess);

    fprintf('MD2_chex: scoring %d images (latent-space MD)...\n', M);

    cnnLatent = extractLatentFeatures(cnnNet, imds, 'relu5');
    reset(imds);
    mlpLatent = extractLatentFeatures(mlpNet, imds, 'relu4');

    [cnnLatConf, ~] = scoreManifold(latentModel.CNN, cnnLatent);
    [mlpLatConf, ~] = scoreManifold(latentModel.MLP, mlpLatent);
    combLatConf     = (cnnLatConf + mlpLatConf) / 2;

    % AND: both networks must independently clear vigilance3
    latentAccepted   = (cnnLatConf >= vigilance3) & (mlpLatConf >= vigilance3);
    latAcceptedCount = sum(latentAccepted);
    latRejectedCount = M - latAcceptedCount;

    fprintf('\n--- Stage 3: latent-space filter ---\n');
    fprintf('  Vigilance3  : %.2f  (AND across CNN and MLP)\n', vigilance3);
    fprintf('  Accepted    : %d (%.1f%% of scored)\n', latAcceptedCount, 100 * latAcceptedCount / M);
    fprintf('  Rejected    : %d (%.1f%% of scored)\n', latRejectedCount, 100 * latRejectedCount / M);
    fprintf('  CNN  latent : mean=%.4f  std=%.4f\n', mean(cnnLatConf), std(cnnLatConf));
    fprintf('  MLP  latent : mean=%.4f  std=%.4f\n', mean(mlpLatConf), std(mlpLatConf));

    results = struct();
    results.NumSamples          = M;
    results.Vigilance3          = vigilance3;
    results.LatentModel         = latentModel;
    results.CNNLatentScores     = cnnLatConf;
    results.MLPLatentScores     = mlpLatConf;
    results.LatentScores        = combLatConf;
    results.LatentAccepted      = latentAccepted;
    results.LatentAcceptedCount = latAcceptedCount;
    results.LatentRejectedCount = latRejectedCount;
end

% ==========================================================================
% Latent manifold: load from cache or build from training images
% ==========================================================================

function latentModel = loadOrTrainLatentManifold(chexRoot, cnnNet, mlpNet, forceRetrain)
    cacheFile = getLatentCacheFile();
    if ~forceRetrain && isfile(cacheFile)
        data = load(cacheFile, 'latentModel', 'cacheMeta');
        if isfield(data, 'latentModel') && isfield(data, 'cacheMeta') && ...
                isfield(data.cacheMeta, 'chexRoot') && strcmp(data.cacheMeta.chexRoot, chexRoot)
            latentModel = data.latentModel;
            fprintf('MD2_chex: loaded cached latent manifold from %s\n', cacheFile);
            return;
        end
    end

    fprintf('MD2_chex: building latent-space manifolds from %s\n', chexRoot);
    imds = imageDatastore(chexRoot, ...
        'FileExtensions', {'.jpg', '.jpeg'}, ...
        'ReadFcn', @readAndPreprocess);

    fprintf('MD2_chex: extracting CNN latent features (relu5, 256-d) from %d images...\n', numel(imds.Files));
    cnnFeatures = extractLatentFeatures(cnnNet, imds, 'relu5');

    reset(imds);
    fprintf('MD2_chex: extracting MLP latent features (relu4, 128-d) from %d images...\n', numel(imds.Files));
    mlpFeatures = extractLatentFeatures(mlpNet, imds, 'relu4');

    latentModel.CNN = trainManifold(cnnFeatures);
    latentModel.MLP = trainManifold(mlpFeatures);

    cacheMeta.chexRoot = chexRoot;
    cacheMeta.savedAt  = char(datetime('now'));
    save(cacheFile, 'latentModel', 'cacheMeta', '-v7.3');
    fprintf('MD2_chex: saved latent manifold to %s\n', cacheFile);
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
% Extract penultimate-layer activations. Returns N x D single matrix.
% D = 256 for relu5 (CNN), 128 for relu4 (MLP).
    reset(imds);
    features = activations(net, imds, layerName, ...
        'MiniBatchSize', 32, 'OutputAs', 'rows');
    features = single(features);
end

% ==========================================================================
% Manifold training and scoring
% ==========================================================================

function model = trainManifold(XTrain)
% Build a single-class PCA manifold from training feature vectors.
% Uses svds for the top NUM_COMPONENTS principal directions only.
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
% Helpers
% ==========================================================================

function img = readAndPreprocess(filename)
% Read a CheXpert JPEG, ensure single-channel [320x390x1] single in [0,1].
    img = imread(filename);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2single(img);
    if ismatrix(img)
        img = reshape(img, size(img, 1), size(img, 2), 1);
    end
end
