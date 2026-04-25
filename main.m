%% main.m
% Full OOD detection pipeline on MNIST using Mahalanobis distance.
%
% Workflow
% --------
%   1. Load MNIST from local IDX files (edit CONFIG section below)
%   2. Train a 4-layer MLP with a bottleneck latent layer
%   3. Extract latent features and apply PCA dimensionality reduction
%   4. Fit per-class Gaussian statistics (means + shared covariance)
%   5. Compute Mahalanobis OOD scores for ID test set and OOD set
%   6. Evaluate: AUROC, AUPR, FPR@95TPR, threshold, OOD count/percentage
%   7. Visualise: latent space (PCA + t-SNE) and score distributions
%
% References
% ----------
%   Lee et al., NeurIPS 2018
%   Venkataramanan et al., ICCVW 2023
%   Anthony & Kamnitsas, arXiv 2309.01488

clear; clc; close all;
rng(42);   % reproducibility

%% =====================================================================
%  CONFIG  –  edit this section to match your setup
% ======================================================================

% --- OOD digits (only used if USE_SEPARATE_OOD = false in loadMNIST) --
OOD_DIGITS = [8, 9];

% --- MLP training settings --------------------------------------------
TRAIN_OPTS = struct( ...
    'latentDim',  128,   ...
    'maxEpochs',  20,    ...
    'miniBatch',  256,   ...
    'learnRate',  1e-3,  ...
    'valFrac',    0.1,   ...
    'verbose',    true);

% --- PCA settings -----------------------------------------------------
PCA_OPTS = struct('varThresh', 0.95);

% --- OOD threshold strategy -------------------------------------------
%   'tpr'      : threshold set so that TPR on ID = THRESHOLD_TPR_TARGET
%                (most common in literature, e.g. 95% TPR)
%   'percentile: threshold set at THRESHOLD_PERCENTILE of ID scores
%                (e.g. 95th percentile of ID scores → 5% ID flagged)
THRESHOLD_MODE       = 'tpr';        % 'tpr' or 'percentile'
THRESHOLD_TPR_TARGET = 0.95;         % used when THRESHOLD_MODE = 'tpr'
THRESHOLD_PERCENTILE = 95;           % used when THRESHOLD_MODE = 'percentile'

% ======================================================================

%% -----------------------------------------------------------------------
%  1.  Load data
% -----------------------------------------------------------------------
fprintf('\n=== Step 1: Loading MNIST ===\n');
[XTrain, YTrain, XTest, YTest, XOod, YOod] = loadMNIST(OOD_DIGITS);

nTrain = size(XTrain, 1);
nTest  = size(XTest,  1);
nOod   = size(XOod,   1);
nTotal = nTest + nOod;

%% -----------------------------------------------------------------------
%  2.  Train MLP
% -----------------------------------------------------------------------
fprintf('\n=== Step 2: Training MLP ===\n');
net = trainMLP(XTrain, YTrain, TRAIN_OPTS);

%% -----------------------------------------------------------------------
%  3.  Extract latent features
% -----------------------------------------------------------------------
fprintf('\n=== Step 3: Extracting Latent Features ===\n');
[ZTrain, pcaModel] = getLatentFeatures(net, XTrain, [],       PCA_OPTS);
[ZTest,  ~       ] = getLatentFeatures(net, XTest,  pcaModel, PCA_OPTS);
[ZOod,   ~       ] = getLatentFeatures(net, XOod,   pcaModel, PCA_OPTS);

%% -----------------------------------------------------------------------
%  4.  Fit Mahalanobis statistics on training data
% -----------------------------------------------------------------------
fprintf('\n=== Step 4: Computing Mahalanobis Statistics ===\n');
[~, mdStats] = computeMahalanobis(ZTrain, YTrain, ZTrain);

%% -----------------------------------------------------------------------
%  5.  Compute OOD scores
% -----------------------------------------------------------------------
fprintf('\n=== Step 5: Computing OOD Scores ===\n');
scoresId  = computeMahalanobis([], [], ZTest, mdStats);
scoresOod = computeMahalanobis([], [], ZOod,  mdStats);

%% -----------------------------------------------------------------------
%  6.  Evaluate OOD detection
% -----------------------------------------------------------------------
fprintf('\n=== Step 6: Evaluating OOD Detection ===\n');

% Ground-truth: 0 = ID, 1 = OOD
allScores = [scoresId;  scoresOod];
allLabels = [zeros(nTest, 1); ones(nOod, 1)];

% --- AUROC ---
[fpr_r, tpr_r, thresholds, AUROC] = perfcurve(allLabels, allScores, 1, ...
                                               'XCrit', 'fpr', 'YCrit', 'tpr');

% --- AUPR (positive = OOD) ---
[~, ~, ~, AUPR] = perfcurve(allLabels, allScores, 1, ...
                             'XCrit', 'reca', 'YCrit', 'prec');

% --- FPR at THRESHOLD_TPR_TARGET ---
[~, tprIdx] = min(abs(tpr_r - THRESHOLD_TPR_TARGET));
FPR95       = fpr_r(tprIdx);

% --- Determine classification threshold ---
switch THRESHOLD_MODE
    case 'tpr'
        % Threshold where TPR on ID side = THRESHOLD_TPR_TARGET
        % i.e. (1 - THRESHOLD_TPR_TARGET) of ID samples are flagged as OOD
        threshold = thresholds(tprIdx);
        threshDesc = sprintf('TPR=%.0f%% on ID set', THRESHOLD_TPR_TARGET*100);

    case 'percentile'
        % Threshold at given percentile of ID scores
        threshold = prctile(scoresId, THRESHOLD_PERCENTILE);
        threshDesc = sprintf('%dth percentile of ID scores', THRESHOLD_PERCENTILE);

    otherwise
        error('Unknown THRESHOLD_MODE: %s. Use ''tpr'' or ''percentile''.', ...
              THRESHOLD_MODE);
end

% --- Apply threshold to get binary OOD predictions ---
predId  = scoresId  > threshold;   % 1 = flagged as OOD
predOod = scoresOod > threshold;   % 1 = correctly flagged as OOD

% Counts
nId_flagged     = sum(predId);          % ID samples wrongly flagged as OOD (FP)
nOod_detected   = sum(predOod);         % OOD samples correctly detected   (TP)
nOod_missed     = sum(~predOod);        % OOD samples missed               (FN)

pct_id_flagged  = 100 * nId_flagged  / nTest;
pct_detected    = 100 * nOod_detected / nOod;
pct_missed      = 100 * nOod_missed   / nOod;
pct_ood_of_total = 100 * nOod / nTotal;

% --- ID classification accuracy ---
YPredTest = classify(net, XTest, 'MiniBatchSize', 512);
idAcc     = mean(YPredTest == YTest) * 100;

% --- Print results ---
fprintf('\n+------------------------------------------------------+\n');
fprintf('|              OOD Detection Results                   |\n');
fprintf('+------------------------------------------------------+\n');
fprintf('  Dataset split\n');
fprintf('    Training samples (ID)       : %d\n',       nTrain);
fprintf('    Test samples     (ID)       : %d\n',       nTest);
fprintf('    OOD  samples                : %d  (%.1f%% of test+OOD)\n', ...
        nOod, pct_ood_of_total);
fprintf('\n  Classification\n');
fprintf('    ID test accuracy            : %.2f%%\n',   idAcc);
fprintf('\n  OOD Detection Metrics\n');
fprintf('    AUROC                       : %.4f\n',     AUROC);
fprintf('    AUPR                        : %.4f\n',     AUPR);
fprintf('    FPR @ %.0f%% TPR             : %.4f\n',    THRESHOLD_TPR_TARGET*100, FPR95);
fprintf('\n  Threshold\n');
fprintf('    Mode                        : %s\n',       THRESHOLD_MODE);
fprintf('    Threshold value             : %.4f\n',     threshold);
fprintf('    Threshold criterion         : %s\n',       threshDesc);
fprintf('\n  At this threshold\n');
fprintf('    OOD detected  (TP)          : %d / %d  (%.1f%%)\n', ...
        nOod_detected, nOod, pct_detected);
fprintf('    OOD missed    (FN)          : %d / %d  (%.1f%%)\n', ...
        nOod_missed,   nOod, pct_missed);
fprintf('    ID flagged as OOD (FP)      : %d / %d  (%.1f%%)\n', ...
        nId_flagged,   nTest, pct_id_flagged);
fprintf('+------------------------------------------------------+\n');

%% -----------------------------------------------------------------------
%  7.  Visualisation
% -----------------------------------------------------------------------
fprintf('\n=== Step 7: Visualising Results ===\n');

% --- 7a. Latent space (PCA + t-SNE) ---
plotLatentFeatures(ZTest, YTest, ZOod, YOod, ...
    sprintf('MNIST Latent Space  (OOD: digits %s)', num2str(OOD_DIGITS)));

% --- 7b. Score distributions and ROC curve ---
figure('Name', 'OOD Detection Summary', 'Color', 'w', ...
       'Position', [150 150 1100 420]);

% Score histograms
subplot(1, 2, 1);
edges = linspace(0, prctile(allScores, 99), 80);
histogram(scoresId,  edges, 'FaceColor', [0.20 0.50 0.80], ...
          'FaceAlpha', 0.65, 'Normalization', 'probability');
hold on;
histogram(scoresOod, edges, 'FaceColor', [0.90 0.25 0.25], ...
          'FaceAlpha', 0.65, 'Normalization', 'probability');
xline(threshold, 'k--', 'LineWidth', 1.8, ...
      'Label', sprintf('Threshold = %.2f', threshold), ...
      'LabelVerticalAlignment', 'bottom');
legend('In-Distribution (ID)', 'Out-of-Distribution (OOD)', ...
       'Location', 'northeast');
xlabel('Mahalanobis Distance Score');
ylabel('Probability');
title(sprintf('Score Distribution\n(OOD detected: %.1f%% | ID flagged: %.1f%%)', ...
              pct_detected, pct_id_flagged));
box on; grid on;

% ROC curve
subplot(1, 2, 2);
plot(fpr_r, tpr_r, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
plot(FPR95, THRESHOLD_TPR_TARGET, 'ro', 'MarkerSize', 8, ...
     'MarkerFaceColor', 'r', 'DisplayName', ...
     sprintf('FPR@%.0f%%TPR = %.4f', THRESHOLD_TPR_TARGET*100, FPR95));
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve  (AUROC = %.4f)', AUROC));
legend('Mahalanobis MD', 'Random', ...
       sprintf('FPR@%.0f%%TPR', THRESHOLD_TPR_TARGET*100), ...
       'Location', 'southeast');
box on; grid on;

sgtitle('Mahalanobis Distance OOD Detection – MNIST', ...
        'FontWeight', 'bold', 'FontSize', 13);

fprintf('\n[main] Pipeline complete.\n');