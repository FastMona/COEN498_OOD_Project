function results = MD_chex(testInput, chexRoot, forceRetrain)
% MD_chex  Wrapper: load CheXpert CNN and MLP models, score a test set.
%
%   NOTE: This is a pass-through wrapper.  Anomaly filtering logic will be
%   added in a future iteration.  For now it loads (or trains) the CNN_chex
%   and MLP_chex models, runs both on the supplied test data, and returns
%   combined anomaly scores.
%
%   RESULTS = MD_chex(testInput)
%     testInput : path to a flat folder of 390x320 .jpg files
%                 OR an imageDatastore of pre-loaded images
%
%   RESULTS = MD_chex(testInput, chexRoot)
%     chexRoot  : path to the 390x320 training image folder (chex_train).
%                 Defaults to .\chex_train relative to this file.
%
%   RESULTS = MD_chex(testInput, chexRoot, forceRetrain)
%     forceRetrain : pass true to ignore cached models and retrain.
%
%   Output struct fields:
%     .CNN            – CNN_chex results struct (trained on chexRoot)
%     .MLP            – MLP_chex results struct (trained on chexRoot)
%     .CNNScores      – Nx1 CNN anomaly scores for testInput
%     .MLPScores      – Nx1 MLP anomaly scores for testInput
%     .CombinedScores – Nx1 mean of CNN and MLP scores
%     .TestFiles      – file paths (if testInput was a folder)
%     .NumSamples     – number of test images scored
%     .MeanCNN        – mean CNN score
%     .MeanMLP        – mean MLP score
%     .MeanCombined   – mean combined score

    if nargin < 1 || isempty(testInput)
        error('MD_chex:noInput', 'Provide a test folder path or imageDatastore.');
    end
    if nargin < 2 || isempty(chexRoot)
        here = fileparts(mfilename('fullpath'));
        chexRoot = fullfile(here, 'chex_train');
    end
    if nargin < 3
        forceRetrain = false;
    end

    % -----------------------------------------------------------------------
    % Ensure models are trained / loaded
    % -----------------------------------------------------------------------
    fprintf('MD_chex: loading CNN model...\n');
    cnnResults = CNN_chex(chexRoot, forceRetrain);

    fprintf('MD_chex: loading MLP model...\n');
    mlpResults = MLP_chex(chexRoot, forceRetrain);

    cnnNet = cnnResults.Network;
    mlpNet = mlpResults.Network;

    % -----------------------------------------------------------------------
    % Prepare test datastore
    % -----------------------------------------------------------------------
    [testImds, testFiles] = prepareTestInput(testInput);
    numSamples = numel(testImds.Files);
    fprintf('MD_chex: scoring %d test images...\n', numSamples);

    % -----------------------------------------------------------------------
    % Inference  (both networks share the same [320 390 1] image format)
    % -----------------------------------------------------------------------
    cnnScores = predict(cnnNet, testImds, 'MiniBatchSize', 32);
    mlpScores = predict(mlpNet, testImds, 'MiniBatchSize', 64);

    combinedScores = (cnnScores + mlpScores) / 2;

    fprintf('\nMD_chex summary\n');
    fprintf('  Test images   : %d\n',   numSamples);
    fprintf('  CNN  score    : mean=%.4f  std=%.4f\n', mean(cnnScores),      std(cnnScores));
    fprintf('  MLP  score    : mean=%.4f  std=%.4f\n', mean(mlpScores),      std(mlpScores));
    fprintf('  Combined score: mean=%.4f  std=%.4f\n', mean(combinedScores), std(combinedScores));
    fprintf('  (normal target = 1.0;  lower values suggest OOD)\n');

    results = struct();
    results.CNN            = cnnResults;
    results.MLP            = mlpResults;
    results.CNNScores      = cnnScores;
    results.MLPScores      = mlpScores;
    results.CombinedScores = combinedScores;
    results.TestFiles      = testFiles;
    results.NumSamples     = numSamples;
    results.MeanCNN        = mean(cnnScores);
    results.MeanMLP        = mean(mlpScores);
    results.MeanCombined   = mean(combinedScores);
end

% ==========================================================================
% Helpers
% ==========================================================================

function [imds, files] = prepareTestInput(testInput)
% Accept a folder path string or an existing imageDatastore.
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
