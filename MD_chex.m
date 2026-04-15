function results = MD_chex(testInput, chexRoot, vigilance, forceRetrain, vigilance3)
% MD_chex  OOD detector for CheXpert chest X-rays — two independent tracks.
%
%   Stage 1 (shared) – Pixel-space MD prefilter
%     Builds a PCA manifold from raw training pixels and rejects inputs
%     whose Mahalanobis confidence falls below vigilance.
%
%   CNN track (independent, Stages 2+3)
%     Stage 2 – CNN regression score  (normal target = 1.0)
%     Stage 3 – CNN latent-space MD   (relu5 features vs normal manifold)
%     Accepted when CNN latent confidence >= vigilance3.
%
%   MLP track (independent, Stages 2+3)
%     Stage 2 – MLP regression score  (normal target = 1.0)
%     Stage 3 – MLP latent-space MD   (relu4 features vs normal manifold)
%     Accepted when MLP latent confidence >= vigilance3.
%
%   The two tracks are completely independent after Stage 1 — no combining,
%   averaging, or AND/OR logic between them.
%
%   RESULTS = MD_chex(testInput)
%   RESULTS = MD_chex(testInput, chexRoot)
%   RESULTS = MD_chex(testInput, chexRoot, vigilance)
%   RESULTS = MD_chex(testInput, chexRoot, vigilance, forceRetrain)
%   RESULTS = MD_chex(testInput, chexRoot, vigilance, forceRetrain, vigilance3)
%
%   Output struct fields:
%     .Vigilance   .Vigilance3   .NumSamples   .TestFiles
%
%     .Stage1
%       .MDScores        Nx1 pixel-MD confidence (all samples)
%       .NormalizedDist  Nx1 normalised Mahalanobis distance
%       .Accepted        Nx1 logical
%       .Rejected        Nx1 logical
%       .AcceptedCount   .RejectedCount
%
%     .CNN  (NaN / false for Stage-1 rejects)
%       .Stage2Scores    Nx1 CNN regression scores
%       .LatentScores    Nx1 CNN latent-MD confidence
%       .Accepted        Nx1 logical: passed S1 AND CNN latent >= vigilance3
%       .AcceptedCount   .RejectedCount
%
%     .MLP  (NaN / false for Stage-1 rejects)
%       .Stage2Scores    Nx1 MLP regression scores
%       .LatentScores    Nx1 MLP latent-MD confidence
%       .Accepted        Nx1 logical: passed S1 AND MLP latent >= vigilance3
%       .AcceptedCount   .RejectedCount

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
    if nargin < 5 || isempty(vigilance3)
        vigilance3 = 0.5;
    end
    if vigilance < 0 || vigilance > 1
        error('MD_chex:badVigilance', 'vigilance must be between 0 and 1.');
    end
    if vigilance3 < 0 || vigilance3 > 1
        error('MD_chex:badVigilance3', 'vigilance3 must be in [0, 1].');
    end

    % -----------------------------------------------------------------------
    % Load / train shared resources
    % -----------------------------------------------------------------------
    pixelModel = loadOrTrainPixelManifold(chexRoot, forceRetrain);

    fprintf('MD_chex: loading CNN model...\n');
    cnnResults = CNN_chex(chexRoot, forceRetrain);
    fprintf('MD_chex: loading MLP model...\n');
    mlpResults = MLP_chex(chexRoot, forceRetrain);
    cnnNet = cnnResults.Network;
    mlpNet = mlpResults.Network;

    latentModel = loadOrTrainLatentManifold(chexRoot, cnnNet, mlpNet, forceRetrain);

    % -----------------------------------------------------------------------
    % Prepare test datastore
    % -----------------------------------------------------------------------
    [testImds, testFiles] = prepareTestInput(testInput);
    numSamples = numel(testImds.Files);
    fprintf('MD_chex: scoring %d test images...\n', numSamples);

    % -----------------------------------------------------------------------
    % Stage 1: pixel-space MD prefilter (shared by both tracks)
    % -----------------------------------------------------------------------
    XTest = extractPixelFeatures(testImds);
    [mdScores, normalizedDist] = scoreManifold(pixelModel, XTest);

    s1Accepted      = mdScores >= vigilance;
    s1AcceptedCount = sum(s1Accepted);
    s1RejectedCount = numSamples - s1AcceptedCount;

    fprintf('\n--- Stage 1: pixel-space prefilter (shared) ---\n');
    fprintf('  Vigilance   : %.2f\n', vigilance);
    fprintf('  Test images : %d\n',  numSamples);
    fprintf('  Accepted    : %d (%.1f%%)\n', s1AcceptedCount, 100 * s1AcceptedCount / numSamples);
    fprintf('  Rejected    : %d (%.1f%%)\n', s1RejectedCount, 100 * s1RejectedCount / numSamples);

    % -----------------------------------------------------------------------
    % Parallel Stages 2+3 — CNN track and MLP track (completely independent)
    % -----------------------------------------------------------------------
    cnnStage2Out     = nan(numSamples, 1, 'single');
    cnnLatentOut     = nan(numSamples, 1, 'single');
    cnnFinalAccepted = false(numSamples, 1);
    cnnAccCount = 0;  cnnRejCount = 0;

    mlpStage2Out     = nan(numSamples, 1, 'single');
    mlpLatentOut     = nan(numSamples, 1, 'single');
    mlpFinalAccepted = false(numSamples, 1);
    mlpAccCount = 0;  mlpRejCount = 0;

    if s1AcceptedCount > 0
        acceptedFiles = testFiles(s1Accepted);
        acceptedImds  = imageDatastore(acceptedFiles, 'ReadFcn', @readAndPreprocess);

        % -------------------------------------------------------------------
        % Stage 2: regression scores (both networks, same S1-accepted subset)
        % -------------------------------------------------------------------
        fprintf('\n--- Stage 2: NN regression ---\n');
        cnnPred = predict(cnnNet, acceptedImds, 'MiniBatchSize', 32);
        reset(acceptedImds);
        mlpPred = predict(mlpNet, acceptedImds, 'MiniBatchSize', 64);

        cnnStage2Out(s1Accepted) = cnnPred;
        mlpStage2Out(s1Accepted) = mlpPred;

        fprintf('  CNN score : mean=%.4f  std=%.4f  (normal target = 1.0)\n', ...
            mean(cnnPred), std(cnnPred));
        fprintf('  MLP score : mean=%.4f  std=%.4f\n', mean(mlpPred), std(mlpPred));

        % -------------------------------------------------------------------
        % Stage 3 — CNN track (independent verdict)
        % -------------------------------------------------------------------
        fprintf('\n--- Stage 3 CNN track: CNN latent manifold ---\n');
        reset(acceptedImds);
        cnnLatFeat = extractLatentFeatures(cnnNet, acceptedImds, 'relu5');
        [cnnLatConf, ~] = scoreManifold(latentModel.CNN, cnnLatFeat);
        cnnLatentOut(s1Accepted) = cnnLatConf;

        cnnPassMask = cnnLatConf >= vigilance3;
        cnnFinalAccepted(s1Accepted) = cnnPassMask;
        cnnAccCount = sum(cnnPassMask);
        cnnRejCount = s1AcceptedCount - cnnAccCount;

        fprintf('  Vigilance3   : %.2f  distanceScale=%.4f\n', vigilance3, latentModel.CNN.distanceScale);
        fprintf('  Latent conf  : mean=%.4f  std=%.4f  min=%.4f  max=%.4f\n', ...
            mean(cnnLatConf), std(cnnLatConf), min(cnnLatConf), max(cnnLatConf));
        fprintf('  Accepted     : %d (%.1f%% of S1-accepted)\n', cnnAccCount, 100*cnnAccCount/s1AcceptedCount);
        fprintf('  Rejected     : %d (%.1f%% of S1-accepted)\n', cnnRejCount, 100*cnnRejCount/s1AcceptedCount);

        % -------------------------------------------------------------------
        % Stage 3 — MLP track (independent verdict)
        % -------------------------------------------------------------------
        fprintf('\n--- Stage 3 MLP track: MLP latent manifold ---\n');
        reset(acceptedImds);
        mlpLatFeat = extractLatentFeatures(mlpNet, acceptedImds, 'relu4');
        [mlpLatConf, ~] = scoreManifold(latentModel.MLP, mlpLatFeat);
        mlpLatentOut(s1Accepted) = mlpLatConf;

        mlpPassMask = mlpLatConf >= vigilance3;
        mlpFinalAccepted(s1Accepted) = mlpPassMask;
        mlpAccCount = sum(mlpPassMask);
        mlpRejCount = s1AcceptedCount - mlpAccCount;

        fprintf('  Vigilance3   : %.2f  distanceScale=%.4f\n', vigilance3, latentModel.MLP.distanceScale);
        fprintf('  Latent conf  : mean=%.4f  std=%.4f  min=%.4f  max=%.4f\n', ...
            mean(mlpLatConf), std(mlpLatConf), min(mlpLatConf), max(mlpLatConf));
        fprintf('  Accepted     : %d (%.1f%% of S1-accepted)\n', mlpAccCount, 100*mlpAccCount/s1AcceptedCount);
        fprintf('  Rejected     : %d (%.1f%% of S1-accepted)\n', mlpRejCount, 100*mlpRejCount/s1AcceptedCount);
    else
        fprintf('  All samples rejected at Stage 1 — Stages 2 and 3 not run.\n');
    end

    % -----------------------------------------------------------------------
    % Pack results
    % -----------------------------------------------------------------------
    results = struct();
    results.Vigilance  = vigilance;
    results.Vigilance3 = vigilance3;
    results.NumSamples = numSamples;
    results.TestFiles  = testFiles;

    % Stage 1 — shared pixel prefilter
    results.Stage1.MDScores       = mdScores;
    results.Stage1.NormalizedDist = normalizedDist;
    results.Stage1.Accepted       = s1Accepted;
    results.Stage1.Rejected       = ~s1Accepted;
    results.Stage1.AcceptedCount  = s1AcceptedCount;
    results.Stage1.RejectedCount  = s1RejectedCount;

    % CNN track
    results.CNN.Stage2Scores  = cnnStage2Out;
    results.CNN.LatentScores  = cnnLatentOut;
    results.CNN.Accepted      = cnnFinalAccepted;
    results.CNN.AcceptedCount = cnnAccCount;
    results.CNN.RejectedCount = cnnRejCount;

    % MLP track
    results.MLP.Stage2Scores  = mlpStage2Out;
    results.MLP.LatentScores  = mlpLatentOut;
    results.MLP.Accepted      = mlpFinalAccepted;
    results.MLP.AcceptedCount = mlpAccCount;
    results.MLP.RejectedCount = mlpRejCount;

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
    fprintf('MD_chex: extracting MLP latent features (relu4, 128-d) from %d images...\n', numel(imds.Files));
    mlpFeatures = extractLatentFeatures(mlpNet, imds, 'relu4');

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
% Returns N x D single matrix (D = 256 for relu5/CNN, 128 for relu4/MLP).
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
    if ismatrix(img)
        img = reshape(img, size(img, 1), size(img, 2), 1);
    end
end
