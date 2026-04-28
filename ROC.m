clear; clc; close all;

%% FILE PATHS

modelFolder = 'D:\Microsoft\OneDrive\Desktop\COEN 6331\Project_Final\Models';

multiFile = fullfile(modelFolder, 'cnn_test_results_all_datasets.mat');
lastFile  = fullfile(modelFolder, 'cnn_test_results_lastlayer_md_auroc.mat');

%% CHECK FILES

if ~isfile(multiFile)
    error('Multi-layer result file not found: %s', multiFile);
end

if ~isfile(lastFile)
    error('Last-layer result file not found: %s', lastFile);
end

%% LOAD RESULTS

fprintf('Loading multi-layer results...\n');
S1 = load(multiFile);

fprintf('Loading last-layer results...\n');
S2 = load(lastFile);

% Extract ROC data
rocX_multi = S1.rocX;
rocY_multi = S1.rocY;
auc_multi  = S1.oodAUROC;

rocX_last = S2.rocX;
rocY_last = S2.rocY;
auc_last  = S2.oodAUROC;

%% PLOT BOTH ROC CURVES

figure('Name','Comparison of ROC Curves: Multi-layer vs Last-layer MD', ...
       'NumberTitle','off');

plot(rocX_multi, rocY_multi, 'LineWidth', 2); hold on;
plot(rocX_last,  rocY_last,  '--', 'LineWidth', 2);

% Random classifier reference
plot([0 1], [0 1], 'k:', 'LineWidth', 1.5);

grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');

title('OOD ROC Curve Comparison (Mahalanobis Distance)');

legend({ ...
    sprintf('Multi-layer MD (AUROC = %.4f)', auc_multi), ...
    sprintf('Last-layer MD (AUROC = %.4f)', auc_last), ...
    'Random Guess' ...
    }, 'Location', 'southeast');

%% OPTIONAL: BETTER VISUAL

set(gca, 'FontSize', 12);
axis([0 1 0 1]);
axis square;

hold off;

%% PRINT SUMMARY

fprintf('\n========================================\n');
fprintf('ROC COMPARISON SUMMARY\n');
fprintf('========================================\n');

fprintf('Multi-layer MD AUROC : %.4f\n', auc_multi);
fprintf('Last-layer MD AUROC  : %.4f\n', auc_last);

if auc_multi > auc_last
    fprintf('=> Multi-layer MD performs better.\n');
elseif auc_multi < auc_last
    fprintf('=> Last-layer MD performs better.\n');
else
    fprintf('=> Both methods perform equally.\n');
end