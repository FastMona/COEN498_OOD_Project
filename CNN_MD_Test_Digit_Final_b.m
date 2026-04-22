clear; clc; close all;

%% USER SETTINGS

testFolder  = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_Final\MNIST Digit_2';
modelFolder = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_Final\Models';

testImageFile = fullfile(testFolder,  'test-images-idx3-ubyte');
testLabelFile = fullfile(testFolder,  'test-labels-idx1-ubyte');
modelFile     = fullfile(modelFolder, 'mnist_cnn_model_with_multilayer_md.mat');

rng('shuffle');

%% CHECK FILES

if ~isfile(testImageFile)
    error('Test image file not found: %s', testImageFile);
end

if ~isfile(testLabelFile)
    error('Test label file not found: %s', testLabelFile);
end

if ~isfile(modelFile)
    error('Saved model not found: %s\nRun the training script first.', modelFile);
end

%% LOAD MODEL

fprintf('Loading trained model + MD stats...\n');
S = load(modelFile);

if ~isfield(S, 'net')
    error('The model file does not contain variable "net".');
end

if ~isfield(S, 'mdStats')
    error('The model file does not contain variable "mdStats".');
end

net = S.net;
mdStats = S.mdStats;

%% CHECK MODEL FORMAT

if ~isfield(mdStats, 'featureLayers')
    error(['This model does not contain "mdStats.featureLayers". ' ...
           'Please use the multi-layer training script.']);
end

if ~isfield(mdStats, 'fusionStats')
    error(['This model does not contain "mdStats.fusionStats". ' ...
           'Please retrain and save the multi-layer MD model.']);
end

%% LOAD TEST DATA

fprintf('Loading test IDX files...\n');

XTest = loadMNISTImages(testImageFile);
YTest = loadMNISTLabels(testLabelFile);

if isempty(XTest) || isempty(YTest)
    error('Could not load the test dataset.');
end

if size(XTest,4) ~= numel(YTest)
    error('Number of test images and labels does not match.');
end

%% PREPROCESS

XTest = single(XTest) / 255;
YTest = categorical(YTest);

fprintf('Test samples: %d\n', numel(YTest));

%% STANDARD CNN PREDICTIONS

YPredAll = classify(net, XTest);

%% MULTI-LAYER MAHALANOBIS OOD SCORES

fprintf('Computing Mahalanobis OOD scores using fused layers:\n');
disp(mdStats.featureLayers);

FTest = extractMultiLayerFeatures(net, XTest, mdStats.featureLayers, mdStats.fusionStats);
FTest = double(FTest);

numTest = size(FTest,1);
numClasses = numel(mdStats.classes);

mdScores = zeros(numTest,1);
nearestClassIdx = zeros(numTest,1);

for i = 1:numTest
    z = FTest(i, :);
    dists = zeros(numClasses,1);

    for c = 1:numClasses
        diffVec = z - mdStats.classMeans(c, :);
        dists(c) = diffVec * mdStats.invSigma * diffVec';
    end

    [mdScores(i), nearestClassIdx(i)] = min(dists);
end

isOOD = mdScores > mdStats.mdThreshold;

%% SUMMARY

numOOD = sum(isOOD);
oodRate = 100 * numOOD / numTest;

fprintf('\n===== OOD DETECTION RESULTS =====\n');
fprintf('Threshold used: %.4f\n', mdStats.mdThreshold);
fprintf('Detected as OOD: %d / %d (%.2f%%)\n', numOOD, numTest, oodRate);

%% OPTIONAL: DEFINE FINAL LABELS

finalLabels = strings(numTest,1);

for i = 1:numTest
    if isOOD(i)
        finalLabels(i) = "OOD";
    else
        finalLabels(i) = string(YPredAll(i));
    end
end

%% RANDOMLY DISPLAY 25 TEST SAMPLES

numShow = 25;

if numTest < numShow
    numShow = numTest;
end

randIdx = randperm(numTest, numShow);

figure('Name','25 Random Test Samples with Multi-Layer MD-OOD','NumberTitle','off');

for i = 1:numShow
    idx = randIdx(i);
    subplot(5,5,i);

    img = XTest(:,:,1,idx);
    imshow(img, []);

    trueLabel  = string(YTest(idx));
    predLabel  = string(YPredAll(idx));
    finalLabel = finalLabels(idx);
    scoreStr   = sprintf('MD: %.2f', mdScores(idx));

    if isOOD(idx)
        statusStr = 'OOD';
    else
        statusStr = 'Accepted';
    end

    title({
        ['True: ' char(trueLabel)]
        ['CNN: ' char(predLabel)]
        ['Final: ' char(finalLabel)]
        scoreStr
        statusStr
        }, 'FontSize', 8);
end

sgtitle(sprintf('Random %d Japanese Test Samples | OOD Rate = %.2f%%', numShow, oodRate));

%% HISTOGRAM OF MD SCORES

figure('Name','Mahalanobis Score Histogram','NumberTitle','off');
histogram(mdScores, 50);
hold on;
xline(mdStats.mdThreshold, 'r', 'LineWidth', 2);
grid on;
xlabel('Mahalanobis Score');
ylabel('Count');
title('Mahalanobis Scores on Japanese Test Set');
legend('Scores', 'Threshold');

%% SHOW ALL NON-OOD ACCEPTED SAMPLES IN MULTIPLE FIGURES

acceptedIdx = find(~isOOD);

fprintf('\nAccepted as ID: %d / %d\n', numel(acceptedIdx), numTest);

if isempty(acceptedIdx)
    fprintf('No accepted samples. All Japanese samples were rejected as OOD.\n');
else
    numPerFigure = 25;
    numAccepted = numel(acceptedIdx);
    numFigures = ceil(numAccepted / numPerFigure);

    for figNum = 1:numFigures
        figure('Name', sprintf('Accepted Samples %d of %d', figNum, numFigures), ...
               'NumberTitle', 'off');

        startIdx = (figNum - 1) * numPerFigure + 1;
        endIdx   = min(figNum * numPerFigure, numAccepted);

        for k = startIdx:endIdx
            subplotIdx = k - startIdx + 1;
            subplot(5,5,subplotIdx);

            idx = acceptedIdx(k);
            img = XTest(:,:,1,idx);
            imshow(img, []);

            title({
                ['True: ' char(string(YTest(idx)))]
                ['CNN: ' char(string(YPredAll(idx)))]
                sprintf('MD: %.2f', mdScores(idx))
                'Accepted'
                }, 'FontSize', 8);
        end

        sgtitle(sprintf('Japanese Samples Accepted as ID | Figure %d of %d', ...
            figNum, numFigures));
    end
end

%% OPTIONAL: BASIC COUNTS OF ACCEPTED DIGIT PREDICTIONS

acceptedPreds = categorical(finalLabels(~isOOD));
if ~isempty(acceptedPreds)
    fprintf('\n===== ACCEPTED DIGIT PREDICTION COUNTS =====\n');
    cats = categories(categorical(string(0:9)));
    for i = 1:numel(cats)
        count_i = sum(acceptedPreds == cats{i});
        fprintf('Digit %s: %d\n', cats{i}, count_i);
    end
end

%% LOCAL FUNCTIONS

function F = extractMultiLayerFeatures(net, X, layerNames, fusionStats)

    numLayers = numel(layerNames);
    featureBlocks = cell(numLayers,1);

    for k = 1:numLayers
        layerName = layerNames{k};
        A = activations(net, X, layerName);

        % Convolutional feature maps: H x W x C x N
        if ndims(A) == 4
            A = squeeze(mean(mean(A,1),2));  % C x N or C x 1

            if isvector(A)
                A = A(:)';   % 1 x C
            else
                A = A';      % N x C
            end
        else
            % FC / vector output
            A = activations(net, X, layerName, 'OutputAs', 'rows'); % N x D
        end

        A = double(A);

        mu = fusionStats(k).mu;
        sigma = fusionStats(k).sigma;
        sigma(sigma < 1e-8) = 1;

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