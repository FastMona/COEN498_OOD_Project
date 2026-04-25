function results = MD1_chex(testInput, chexRoot, vigilance, forceRetrain)
% MD1_chex  Stage 1: pixel-space Mahalanobis Distance prefilter.
%
%   Builds (or loads from cache) a PCA manifold from the pixel vectors of
%   chexRoot training images, then scores each test image against it.
%   Images whose Mahalanobis confidence falls below vigilance are flagged
%   as out-of-distribution at the pixel level ("not a chest X-ray").
%
%   RESULTS = MD1_chex(testInput, chexRoot, vigilance, forceRetrain)
%     testInput    : folder path or imageDatastore of test images
%     chexRoot     : flat folder of 390x320 training .jpg files
%     vigilance    : acceptance confidence in [0,1]  (default 0.5)
%     forceRetrain : true forces manifold rebuild    (default false)
%
%   Output struct fields (all N x 1, one entry per test image):
%     .NumSamples     – total test images
%     .TestFiles      – Nx1 cell of file paths
%     .Vigilance      – threshold used
%     .PixelModel     – PCA manifold struct
%     .MDScores       – Nx1 Mahalanobis confidence (higher = more normal)
%     .NormalizedDist – Nx1 normalised Mahalanobis distance
%     .Accepted       – Nx1 logical: passed Stage 1
%     .IsOOD          – Nx1 logical: rejected at Stage 1
%     .AcceptedCount  – number accepted
%     .RejectedCount  – number rejected

    if nargin < 1 || isempty(testInput)
        error('MD1_chex:noInput', 'Provide a test folder path or imageDatastore.');
    end
    if nargin < 2 || isempty(chexRoot)
        here     = fileparts(mfilename('fullpath'));
        chexRoot = fullfile(here, 'chex_train');
    end
    if nargin < 3 || isempty(vigilance)
        vigilance = 0.5;
    end
    if nargin < 4
        forceRetrain = false;
    end
    if vigilance < 0 || vigilance > 1
        error('MD1_chex:badVigilance', 'vigilance must be in [0, 1].');
    end

    % -----------------------------------------------------------------------
    % Load or build pixel-space PCA manifold
    % -----------------------------------------------------------------------
    pixelModel = loadOrTrainPixelManifold(chexRoot, forceRetrain);

    % -----------------------------------------------------------------------
    % Prepare test datastore
    % -----------------------------------------------------------------------
    [testImds, testFiles] = prepareTestInput(testInput);
    numSamples = numel(testImds.Files);
    fprintf('MD1_chex: scoring %d test images (pixel-space MD)...\n', numSamples);

    % -----------------------------------------------------------------------
    % Score all test images
    % -----------------------------------------------------------------------
    XTest = extractPixelFeatures(testImds);
    [mdScores, normalizedDist] = scoreManifold(pixelModel, XTest);

    accepted      = mdScores >= vigilance;
    acceptedCount = sum(accepted);
    rejectedCount = numSamples - acceptedCount;

    results = struct();
    results.NumSamples     = numSamples;
    results.TestFiles      = testFiles;
    results.Vigilance      = vigilance;
    results.PixelModel     = pixelModel;
    results.MDScores       = mdScores;
    results.NormalizedDist = normalizedDist;
    results.Accepted       = accepted;
    results.IsOOD          = ~accepted;
    results.AcceptedCount  = acceptedCount;
    results.RejectedCount  = rejectedCount;
end

% ==========================================================================
% Pixel manifold: load from cache or build from training images
% ==========================================================================

function model = loadOrTrainPixelManifold(chexRoot, forceRetrain)
    cacheFile = getPixelCacheFile();
    if ~forceRetrain && isfile(cacheFile)
        data = load(cacheFile, 'model', 'cacheMeta');
        if isfield(data, 'model') && isfield(data, 'cacheMeta') && ...
                isfield(data.cacheMeta, 'chexRoot') && strcmp(data.cacheMeta.chexRoot, chexRoot)
            model = data.model;
            fprintf('MD1_chex: loaded cached pixel manifold from %s\n', cacheFile);
            return;
        end
    end

    fprintf('MD1_chex: building pixel-space PCA manifold from %s\n', chexRoot);
    imds = imageDatastore(chexRoot, ...
        'FileExtensions', {'.jpg', '.jpeg'}, ...
        'ReadFcn', @readAndPreprocess);
    if numel(imds.Files) == 0
        error('MD1_chex:noTrainImages', 'No .jpg files found in chexRoot: %s', chexRoot);
    end

    fprintf('MD1_chex: extracting pixel features from %d training images...\n', numel(imds.Files));
    XTrain = extractPixelFeatures(imds);
    model  = trainManifold(XTrain);

    cacheMeta.chexRoot = chexRoot;
    cacheMeta.savedAt  = char(datetime('now'));
    save(cacheFile, 'model', 'cacheMeta', '-v7.3');
    fprintf('MD1_chex: saved pixel manifold to %s\n', cacheFile);
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

function [imds, files] = prepareTestInput(testInput)
    if ischar(testInput) || (isstring(testInput) && isscalar(testInput))
        folderPath = char(string(testInput));
        if ~isfolder(folderPath)
            error('MD1_chex:missingFolder', 'Test folder not found: %s', folderPath);
        end
        imds  = imageDatastore(folderPath, ...
            'FileExtensions', {'.jpg', '.jpeg'}, ...
            'ReadFcn', @readAndPreprocess);
        files = imds.Files;
    elseif isa(testInput, 'matlab.io.datastore.ImageDatastore')
        imds  = testInput;
        files = imds.Files;
    else
        error('MD1_chex:unsupportedInput', ...
            'testInput must be a folder path string or an imageDatastore.');
    end
    if numel(imds.Files) == 0
        error('MD1_chex:noImages', 'No .jpg files found in the test input.');
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
