clear; clc; close all;

%% USER SETTINGS

testFolder  = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_Final\MNIST Digit_2';
modelFolder = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_Final\Models';

testImageFile = fullfile(testFolder,  'test-images-idx3-ubyte');
testLabelFile = fullfile(testFolder,  'test-labels-idx1-ubyte');
modelFile     = fullfile(modelFolder, 'mnist_cnn_model_with_md.mat');

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

%% MAHALANOBIS OOD SCORES

fprintf('Computing Mahalanobis OOD scores using layer: %s\n', mdStats.featureLayer);

FTest = activations(net, XTest, mdStats.featureLayer, 'OutputAs', 'rows');
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
    error('Test set has fewer than 25 samples.');
end

randIdx = randperm(numTest, numShow);

figure('Name','25 Random Test Samples with MD-OOD','NumberTitle','off');

for i = 1:numShow
    idx = randIdx(i);
    subplot(5,5,i);

    img = XTest(:,:,1,idx);
    imshow(img, []);

    trueLabel = string(YTest(idx));
    predLabel = string(YPredAll(idx));
    finalLabel = finalLabels(idx);
    scoreStr = sprintf('MD: %.2f', mdScores(idx));

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

sgtitle(sprintf('Random 25 Japanese Test Samples | OOD Rate = %.2f%%', oodRate));

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

% acceptedIdx = find(~isOOD);
% 
% fprintf('\nAccepted as ID: %d / %d\n', numel(acceptedIdx), numTest);
% 
% if isempty(acceptedIdx)
%     fprintf('No accepted samples. All Japanese samples were rejected as OOD.\n');
% else
%     numPerFigure = 25;
%     numAccepted = numel(acceptedIdx);
%     numFigures = ceil(numAccepted / numPerFigure);
% 
%     for figNum = 1:numFigures
%         figure('Name', sprintf('Accepted Samples %d of %d', figNum, numFigures), ...
%                'NumberTitle', 'off');
% 
%         startIdx = (figNum - 1) * numPerFigure + 1;
%         endIdx = min(figNum * numPerFigure, numAccepted);
% 
%         for k = startIdx:endIdx
%             subplotIdx = k - startIdx + 1;
%             subplot(5,5,subplotIdx);
% 
%             idx = acceptedIdx(k);
%             img = XTest(:,:,1,idx);
%             imshow(img, []);
% 
%             title({
%                 ['True: ' char(string(YTest(idx)))]
%                 ['CNN: ' char(string(YPredAll(idx)))]
%                 sprintf('MD: %.2f', mdScores(idx))
%                 'Accepted'
%                 }, 'FontSize', 8);
%         end
% 
%         sgtitle(sprintf('Japanese Samples Accepted as ID | Figure %d of %d', ...
%             figNum, numFigures));
%     end
% end

%% LOCAL FUNCTIONS

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