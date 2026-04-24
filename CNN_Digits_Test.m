clear; clc; close all;

%% USER SETTINGS

here        = fileparts(mfilename('fullpath'));
trainFolder = askFolder(fullfile(here, 'MNIST_digits', 'raw'), 'Training folder');
testFolder  = askFolder(fullfile(here, 'KMNIST_japanese'),     'Test folder');

modelFile = fullfile(trainFolder, 'mnist_cnn_model.mat');

rng('shuffle');

%% CHECK MODEL

if ~isfile(modelFile)
    error('Saved model not found: %s\nRun CNN_Digits_Train.m first.', modelFile);
end

%% LOAD MODEL

fprintf('Loading trained model...\n');
S = load(modelFile);

if ~isfield(S, 'net')
    error('The model file does not contain variable "net".');
end

net = S.net;

%% DETECT FORMAT AND LOAD TEST DATA

idxImagesFile = fullfile(testFolder, 't10k-images-idx3-ubyte');
idxLabelsFile = fullfile(testFolder, 't10k-labels-idx1-ubyte');
csvFile       = fullfile(testFolder, 'mnist_test.csv');

if isfile(idxImagesFile) && isfile(idxLabelsFile)
    fprintf('Detected IDX format. Loading from: %s\n', testFolder);
    [XTest, YTest] = loadIDX(idxImagesFile, idxLabelsFile);
elseif isfile(csvFile)
    fprintf('Detected CSV format. Loading from: %s\n', csvFile);
    [XTest, YTest] = loadCSV(csvFile);
else
    error(['No recognised test data found in:\n  %s\n' ...
           'Expected ''t10k-images-idx3-ubyte'' + ''t10k-labels-idx1-ubyte'' (IDX) ' ...
           'or ''mnist_test.csv'' (CSV).'], testFolder);
end

fprintf('Test samples: %d\n', numel(YTest));

%% FULL TEST ACCURACY

YPredAll = classify(net, XTest);
fullAccuracy = mean(YPredAll == YTest);

fprintf('\n===== FULL MNIST TEST SET =====\n');
fprintf('Full test accuracy: %.2f %%\n', 100*fullAccuracy);

%% CONFUSION MATRIX

figure('Name','Confusion Matrix','NumberTitle','off');
confusionchart(YTest, YPredAll);
title(sprintf('Confusion Matrix | Full Test Accuracy = %.2f%%', 100*fullAccuracy));

%% RANDOMLY PICK 25 TEST SAMPLES

numShow = 25;
numTest = numel(YTest);

if numTest < numShow
    error('Test set has fewer than 25 samples.');
end

randIdx = randperm(numTest, numShow);

XTest25 = XTest(:,:,:,randIdx);
YTest25 = YTest(randIdx);

%% PREDICT 25 SAMPLES

YPred25 = classify(net, XTest25);
accuracy25 = mean(YPred25 == YTest25);

fprintf('\n===== RANDOM 25-SAMPLE TEST RESULTS =====\n');
fprintf('Correct: %d / %d\n', sum(YPred25 == YTest25), numShow);
fprintf('Accuracy: %.2f %%\n', 100*accuracy25);

%% DISPLAY 25 RESULTS

figure('Name','25 Random MNIST Test Samples','NumberTitle','off');

for i = 1:numShow
    subplot(5,5,i);

    img = XTest25(:,:,1,i);
    imshow(img, []);

    trueLabel = string(YTest25(i));
    predLabel = string(YPred25(i));

    if YPred25(i) == YTest25(i)
        resultStr = 'Correct';
    else
        resultStr = 'Wrong';
    end

    title({
        ['True: ' char(trueLabel)]
        ['Pred: ' char(predLabel)]
        resultStr
        }, 'FontSize', 9);
end

sgtitle(sprintf('Random 25 MNIST Test Samples | Accuracy = %.2f%%', 100*accuracy25));

%% SHOW ALL WRONG PREDICTIONS IN MULTIPLE FIGURES

wrongIdx = find(YPredAll ~= YTest);

if isempty(wrongIdx)
    fprintf('\nNo wrong predictions in the test set.\n');
else
    numPerFigure = 25;
    numWrong = numel(wrongIdx);
    numFigures = ceil(numWrong / numPerFigure);

    fprintf('\nTotal wrong predictions: %d\n', numWrong);
    fprintf('Number of figures needed: %d\n', numFigures);

    for figNum = 1:numFigures
        figure('Name', sprintf('Wrong Predictions %d of %d', figNum, numFigures), ...
               'NumberTitle', 'off');

        startIdx = (figNum - 1) * numPerFigure + 1;
        endIdx = min(figNum * numPerFigure, numWrong);

        for k = startIdx:endIdx
            subplotIdx = k - startIdx + 1;
            subplot(5,5,subplotIdx);

            img = XTest(:,:,1,wrongIdx(k));
            imshow(img, []);

            trueLabel = string(YTest(wrongIdx(k)));
            predLabel = string(YPredAll(wrongIdx(k)));

            title({
                ['True: ' char(trueLabel)]
                ['Pred: ' char(predLabel)]
                'Wrong'
                }, 'FontSize', 8);
        end

        sgtitle(sprintf('Wrong Predictions | Figure %d of %d', figNum, numFigures));
    end
end

%% LOCAL FUNCTIONS

function folder = askFolder(defaultPath, label)
    fprintf('%s [%s]:\n', label, defaultPath);
    reply = strtrim(input('? ', 's'));
    if isempty(reply), folder = defaultPath; else, folder = reply; end
end

function [XTest, YTest] = loadCSV(csvFile)
    testData = readmatrix(csvFile);
    if isempty(testData)
        error('Could not read CSV file: %s', csvFile);
    end
    if size(testData, 2) ~= 785
        error('Expected 785 columns (1 label + 784 pixels) in %s, got %d.', csvFile, size(testData,2));
    end
    labels = testData(:, 1);
    pixels = single(testData(:, 2:end)) / 255;
    XTest  = reshape(pixels', 28, 28, 1, []);
    YTest  = categorical(labels, 0:9, string(0:9));
end

function [XTest, YTest] = loadIDX(imagesFile, labelsFile)
    XTest  = readIDXImages(imagesFile);
    labels = readIDXLabels(labelsFile);
    YTest  = categorical(labels, 0:9, string(0:9));
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
