clear; clc; close all;

%% USER SETTINGS

testFolder  = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_Final\Test Sets';
modelFolder = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_Final\Models';

modelFile = fullfile(modelFolder, 'mnist_cnn_model_with_multilayer_md_auroc.mat');

rng('shuffle');

%% DATASETS TO TEST

datasets(1).name = 'Digit';
datasets(1).imageFile = fullfile(testFolder, 'test-images-idx3-ubyte_digit');
datasets(1).labelFile = fullfile(testFolder, 'test-labels-idx1-ubyte_digit');
datasets(1).isID = true;

datasets(2).name = 'Fashion';
datasets(2).imageFile = fullfile(testFolder, 'test-images-idx3-ubyte_fashion');
datasets(2).labelFile = fullfile(testFolder, 'test-labels-idx1-ubyte_fashion');
datasets(2).isID = false;

datasets(3).name = 'Japanese';
datasets(3).imageFile = fullfile(testFolder, 'test-images-idx3-ubyte_japanese');
datasets(3).labelFile = fullfile(testFolder, 'test-labels-idx1-ubyte_japanese');
datasets(3).isID = false;

%% CHECK MODEL FILE

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

if ~isfield(mdStats, 'featureLayers')
    error('This model does not contain mdStats.featureLayers.');
end

if ~isfield(mdStats, 'fusionStats')
    error('This model does not contain mdStats.fusionStats.');
end

%% STORAGE FOR ALL DATASETS

allMDScores = [];
allOODLabels = [];

allMdScoresByDataset = cell(numel(datasets), 1);
allNames = cell(numel(datasets), 1);
mdThresholdGlobal = mdStats.mdThreshold;

summaryResults = struct([]);

%% TEST EACH DATASET

for d = 1:numel(datasets)

    datasetName = datasets(d).name;
    imageFile = datasets(d).imageFile;
    labelFile = datasets(d).labelFile;
    isIDDataset = datasets(d).isID;

    fprintf('\n========================================\n');
    fprintf('Testing dataset: %s\n', datasetName);
    fprintf('========================================\n');

    %% CHECK FILES

    if ~isfile(imageFile)
        error('Image file not found for %s: %s', datasetName, imageFile);
    end

    if ~isfile(labelFile)
        error('Label file not found for %s: %s', datasetName, labelFile);
    end

    %% LOAD DATA

    XTest = loadMNISTImages(imageFile);
    YTest = loadMNISTLabels(labelFile);

    if isempty(XTest) || isempty(YTest)
        error('Could not load dataset: %s', datasetName);
    end

    if size(XTest,4) ~= numel(YTest)
        error('Number of images and labels does not match for %s.', datasetName);
    end

    %% PREPROCESS

    XTest = single(XTest) / 255;
    YTest = categorical(YTest);

    numTest = numel(YTest);

    fprintf('Samples: %d\n', numTest);

    %% CNN PREDICTIONS

    YPredAll = classify(net, XTest);
    YPredScores = predict(net, XTest);

    %% CLASSIFICATION ACCURACY AND AUROC ONLY FOR DIGIT ID DATASET

    accuracy = NaN;
    macroAUROC = NaN;
    aucPerClass = [];

    if isIDDataset

        accuracy = mean(YPredAll == YTest) * 100;
        fprintf('CNN Accuracy on %s: %.2f%%\n', datasetName, accuracy);

        classes = categories(YTest);
        numClasses = numel(classes);

        aucPerClass = zeros(numClasses, 1);

        fprintf('\nComputing CNN classification AUROC on %s...\n', datasetName);

        for c = 1:numClasses
            positiveClass = classes{c};
            binaryLabels = (YTest == positiveClass);
            classScores = YPredScores(:, c);

            [~,~,~,aucPerClass(c)] = perfcurve(binaryLabels, classScores, true);

            fprintf('Class %s AUROC: %.4f\n', positiveClass, aucPerClass(c));
        end

        macroAUROC = mean(aucPerClass);
        fprintf('Macro AUROC on %s: %.4f\n', datasetName, macroAUROC);

    else
        fprintf('Skipping classification accuracy/AUROC for %s because it is OOD.\n', datasetName);
    end

    %% MAHALANOBIS OOD SCORES

    fprintf('\nComputing Mahalanobis OOD scores for %s...\n', datasetName);

    FTest = extractMultiLayerFeatures(net, XTest, mdStats.featureLayers, mdStats.fusionStats);
    FTest = double(FTest);

    numSamples = size(FTest, 1);
    numClassesMD = numel(mdStats.classes);

    mdScores = zeros(numSamples, 1);
    nearestClassIdx = zeros(numSamples, 1);

    for i = 1:numSamples
        z = FTest(i, :);
        dists = zeros(numClassesMD, 1);

        for c = 1:numClassesMD
            diffVec = z - mdStats.classMeans(c, :);
            dists(c) = diffVec * mdStats.invSigma * diffVec';
        end

        [mdScores(i), nearestClassIdx(i)] = min(dists);
    end

    isOOD = mdScores > mdStats.mdThreshold;

    numOOD = sum(isOOD);
    oodRate = 100 * numOOD / numSamples;

    fprintf('\n===== %s OOD DETECTION RESULTS =====\n', datasetName);
    fprintf('Threshold used: %.4f\n', mdStats.mdThreshold);
    fprintf('Detected as OOD: %d / %d (%.2f%%)\n', numOOD, numSamples, oodRate);

    %% STORE FOR OVERALL OOD AUROC AND COMBINED HISTOGRAM

    if isIDDataset
        trueOOD = zeros(numSamples, 1);
    else
        trueOOD = ones(numSamples, 1);
    end

    allMDScores = [allMDScores; mdScores];
    allOODLabels = [allOODLabels; trueOOD];

    allMdScoresByDataset{d} = mdScores;
    allNames{d} = datasetName;

    %% FINAL LABELS

    finalLabels = strings(numSamples, 1);

    for i = 1:numSamples
        if isOOD(i)
            finalLabels(i) = "OOD";
        else
            finalLabels(i) = string(YPredAll(i));
        end
    end

    %% DISPLAY 5 ACCEPTED ID + 5 REJECTED OOD PATTERNS

    idIdx  = find(~isOOD);
    oodIdx = find(isOOD);

    numIDShow  = min(5, numel(idIdx));
    numOODShow = min(5, numel(oodIdx));

    figure('Name', sprintf('%s Accepted ID and Rejected OOD Patterns', datasetName), ...
           'NumberTitle', 'off');

    % ---------- ROW 1: ACCEPTED ID ----------
    if numIDShow > 0
        idSample = idIdx(randperm(numel(idIdx), numIDShow));

        for i = 1:numIDShow
            idx = idSample(i);
            subplot(2,5,i);

            imshow(XTest(:,:,1,idx), []);

            title({
                'Accepted ID'
                ['True: ' char(string(YTest(idx)))]
                ['CNN: ' char(string(YPredAll(idx)))]
                sprintf('MD: %.2f', mdScores(idx))
                }, 'FontSize', 8);
        end
    end

    % ---------- ROW 2: REJECTED OOD ----------
    if numOODShow > 0
        oodSample = oodIdx(randperm(numel(oodIdx), numOODShow));

        for i = 1:numOODShow
            idx = oodSample(i);
            subplot(2,5,5+i);

            imshow(XTest(:,:,1,idx), []);

            title({
                'Rejected OOD'
                ['True: ' char(string(YTest(idx)))]
                ['CNN: ' char(string(YPredAll(idx)))]
                sprintf('MD: %.2f', mdScores(idx))
                }, 'FontSize', 8);
        end
    end

    sgtitle(sprintf('%s Test Set: Accepted vs Rejected', datasetName));

    drawnow;

    %% SAVE SUMMARY

    summaryResults(d).datasetName = datasetName;
    summaryResults(d).numSamples = numSamples;
    summaryResults(d).isID = isIDDataset;
    summaryResults(d).accuracy = accuracy;
    summaryResults(d).macroAUROC = macroAUROC;
    summaryResults(d).aucPerClass = aucPerClass;
    summaryResults(d).numOOD = numOOD;
    summaryResults(d).oodRate = oodRate;
    summaryResults(d).mdScores = mdScores;
    summaryResults(d).isOOD = isOOD;
    summaryResults(d).YPredAll = YPredAll;
    summaryResults(d).YTest = YTest;
    summaryResults(d).finalLabels = finalLabels;

end

%% COMBINED HISTOGRAM OF MD SCORES FOR ALL TEST SETS

figure('Name', 'Multi-Layer MD + AUROC: Mahalanobis Score Histogram for All Test Sets', ...
       'NumberTitle', 'off');

hold on;

for d = 1:numel(datasets)
    histogram(allMdScoresByDataset{d}, 50, ...
        'Normalization', 'probability', ...
        'DisplayName', allNames{d}, ...
        'FaceAlpha', 0.35);
end

xline(mdThresholdGlobal, 'r', 'LineWidth', 2, ...
      'DisplayName', 'OOD Threshold');

grid on;
xlabel('Mahalanobis Score');
ylabel('Probability');
title('Mahalanobis Score Distribution');
legend('show');

hold off;

%% OVERALL OOD AUROC

fprintf('\n========================================\n');
fprintf('OVERALL OOD AUROC USING MAHALANOBIS SCORES\n');
fprintf('========================================\n');

[rocX, rocY, rocT, oodAUROC] = perfcurve(allOODLabels, allMDScores, 1);

fprintf('Overall OOD AUROC: %.4f\n', oodAUROC);

figure('Name','Overall OOD ROC Curve','NumberTitle','off');
plot(rocX, rocY, 'LineWidth', 2);
grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Overall OOD ROC Curve | AUROC = %.4f', oodAUROC));

%% SUMMARY TABLE

fprintf('\n========================================\n');
fprintf('SUMMARY TABLE\n');
fprintf('========================================\n');

fprintf('%-12s %-10s %-12s %-12s %-12s\n', ...
    'Dataset', 'Samples', 'Accuracy', 'MacroAUROC', 'OOD Rate');

for d = 1:numel(summaryResults)

    if isnan(summaryResults(d).accuracy)
        accStr = 'N/A';
    else
        accStr = sprintf('%.2f%%', summaryResults(d).accuracy);
    end

    if isnan(summaryResults(d).macroAUROC)
        aucStr = 'N/A';
    else
        aucStr = sprintf('%.4f', summaryResults(d).macroAUROC);
    end

    fprintf('%-12s %-10d %-12s %-12s %-12.2f%%\n', ...
        summaryResults(d).datasetName, ...
        summaryResults(d).numSamples, ...
        accStr, ...
        aucStr, ...
        summaryResults(d).oodRate);
end

%% SAVE TEST RESULTS

resultFile = fullfile(modelFolder, 'cnn_test_results_all_datasets.mat');

save(resultFile, ...
    'summaryResults', ...
    'allMDScores', ...
    'allOODLabels', ...
    'allMdScoresByDataset', ...
    'allNames', ...
    'oodAUROC', ...
    'rocX', ...
    'rocY', ...
    'rocT', ...
    '-v7.3');

fprintf('\nSaved all test results to:\n%s\n', resultFile);

%% LOCAL FUNCTIONS

function F = extractMultiLayerFeatures(net, X, layerNames, fusionStats)

    numLayers = numel(layerNames);
    featureBlocks = cell(numLayers,1);

    for k = 1:numLayers
        layerName = layerNames{k};
        A = activations(net, X, layerName);

        if ndims(A) == 4
            A = squeeze(mean(mean(A,1),2));

            if isvector(A)
                A = A(:)';
            else
                A = A';
            end
        else
            A = activations(net, X, layerName, 'OutputAs', 'rows');
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