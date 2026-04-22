function results = evaluate_auroc(normalFolder, oodFolder, chexRoot, vigilance, forceRetrain, vigilance3)
% evaluate_auroc  Compute AUROC for every pipeline score against a labelled pair of folders.
%
%   Runs MD_chex on normalFolder (label=0, in-distribution) and oodFolder
%   (label=1, OOD), then computes and plots the ROC curve and AUROC for each
%   of the five pipeline scores independently:
%
%     Stage 1  – pixel-space MD confidence          (all samples)
%     CNN S2   – CNN regression score               (Stage-1-accepted only)
%     CNN lat  – CNN latent-space MD confidence     (Stage-1-accepted only)
%     MLP S2   – MLP regression score               (Stage-1-accepted only)
%     MLP lat  – MLP latent-space MD confidence     (Stage-1-accepted only)
%
%   Stage 2/3 scores are evaluated only on Stage-1-accepted samples because
%   those scores are NaN for Stage-1 rejects.
%
%   Requires Statistics and Machine Learning Toolbox (perfcurve).
%
%   RESULTS = evaluate_auroc(normalFolder, oodFolder)
%   RESULTS = evaluate_auroc(normalFolder, oodFolder, chexRoot)
%   RESULTS = evaluate_auroc(normalFolder, oodFolder, chexRoot, vigilance)
%   RESULTS = evaluate_auroc(normalFolder, oodFolder, chexRoot, vigilance, forceRetrain)
%   RESULTS = evaluate_auroc(normalFolder, oodFolder, chexRoot, vigilance, forceRetrain, vigilance3)
%
%   Output struct:
%     .NormalFolder  .OODFolder  .ChexRoot  .Vigilance  .Vigilance3
%     .Stage1 .CNN.Stage2 .CNN.Latent .MLP.Stage2 .MLP.Latent
%       each sub-struct has: .AUC  .FPR  .TPR
%     .MDNormal  .MDOod  (full MD_chex output structs)

    if nargin < 1 || isempty(normalFolder)
        normalFolder = getSetFolderPaths('resolve', 'aurocNormalRoot');
    else
        normalFolder = getSetFolderPaths('resolve', 'aurocNormalRoot', normalFolder);
    end
    if nargin < 2 || isempty(oodFolder)
        oodFolder = getSetFolderPaths('resolve', 'aurocOodRoot');
    else
        oodFolder = getSetFolderPaths('resolve', 'aurocOodRoot', oodFolder);
    end

    if nargin < 3 || isempty(chexRoot)
        here = fileparts(mfilename('fullpath'));
        chexRoot = fullfile(here, 'chex_train');
    end
    if nargin < 4 || isempty(vigilance)
        vigilance = 0.5;
    end
    if nargin < 5
        forceRetrain = false;
    end
    if nargin < 6 || isempty(vigilance3)
        vigilance3 = 0.5;
    end

    if ~isfolder(normalFolder)
        error('evaluate_auroc:badFolder', 'Normal folder not found: %s', normalFolder);
    end
    if ~isfolder(oodFolder)
        error('evaluate_auroc:badFolder', 'OOD folder not found: %s', oodFolder);
    end

    [~, normName] = fileparts(normalFolder);
    [~, oodName]  = fileparts(oodFolder);

    fprintf('=== evaluate_auroc ===\n');
    fprintf('  Normal (label=0) : %s\n', normalFolder);
    fprintf('  OOD    (label=1) : %s\n', oodFolder);
    fprintf('  Training data    : %s\n', chexRoot);
    fprintf('  Vigilance S1=%.2f  S3=%.2f\n\n', vigilance, vigilance3);

    % -----------------------------------------------------------------------
    % Run MD_chex on both folders (models load from cache on second call)
    % -----------------------------------------------------------------------
    fprintf('--- Scoring normal folder ---\n');
    mdNorm = MD_chex(normalFolder, chexRoot, vigilance, forceRetrain, vigilance3);

    fprintf('\n--- Scoring OOD folder ---\n');
    mdOod  = MD_chex(oodFolder,   chexRoot, vigilance, false,        vigilance3);

    nNorm = mdNorm.NumSamples;
    nOod  = mdOod.NumSamples;

    normS1 = mdNorm.Stage1.Accepted;   % logical mask, normal images passing S1
    oodS1  = mdOod.Stage1.Accepted;    % logical mask, OOD images passing S1

    nNormS1 = sum(normS1);
    nOodS1  = sum(oodS1);

    fprintf('\n  S1-accepted — normal: %d / %d    OOD: %d / %d\n\n', ...
        nNormS1, nNorm, nOodS1, nOod);

    % -----------------------------------------------------------------------
    % Stage 1: all samples
    % -----------------------------------------------------------------------
    s1 = buildROC( ...
        [mdNorm.Stage1.MDScores;  mdOod.Stage1.MDScores], ...
        [zeros(nNorm, 1);         ones(nOod, 1)]);

    % -----------------------------------------------------------------------
    % Stages 2+3: Stage-1-accepted subset only
    % -----------------------------------------------------------------------
    s2Labels = [zeros(nNormS1, 1); ones(nOodS1, 1)];

    cnnS2  = buildROC( ...
        [mdNorm.CNN.Stage2Scores(normS1);  mdOod.CNN.Stage2Scores(oodS1)], ...
        s2Labels);

    cnnLat = buildROC( ...
        [mdNorm.CNN.LatentScores(normS1);  mdOod.CNN.LatentScores(oodS1)], ...
        s2Labels);

    mlpS2  = buildROC( ...
        [mdNorm.MLP.Stage2Scores(normS1);  mdOod.MLP.Stage2Scores(oodS1)], ...
        s2Labels);

    mlpLat = buildROC( ...
        [mdNorm.MLP.LatentScores(normS1);  mdOod.MLP.LatentScores(oodS1)], ...
        s2Labels);

    % -----------------------------------------------------------------------
    % Print summary table
    % -----------------------------------------------------------------------
    fprintf('=== AUROC Summary  |  normal: %s  vs  OOD: %s ===\n', normName, oodName);
    fprintf('  %-32s  AUC\n', 'Score');
    fprintf('  %-32s  -----\n', repmat('-', 1, 32));
    printRow('Stage 1: pixel MD',          s1.AUC);
    printRow('CNN track — Stage 2 regr',   cnnS2.AUC);
    printRow('CNN track — latent MD',      cnnLat.AUC);
    printRow('MLP track — Stage 2 regr',   mlpS2.AUC);
    printRow('MLP track — latent MD',      mlpLat.AUC);
    fprintf('\n');

    % -----------------------------------------------------------------------
    % Figure 1: ROC curves
    % -----------------------------------------------------------------------
    figure('Name', sprintf('AUROC: %s vs %s', normName, oodName), 'Color', 'w');
    hold on;

    plotROC(s1,     'k-',  'Stage 1: pixel MD');
    plotROC(cnnS2,  'b-',  'CNN S2 regression');
    plotROC(cnnLat, 'b--', 'CNN latent MD');
    plotROC(mlpS2,  'g-',  'MLP S2 regression');
    plotROC(mlpLat, 'g--', 'MLP latent MD');
    plot([0 1], [0 1], 'k:', 'LineWidth', 1);

    hold off;
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    grid on;  box on;
    legend( ...
        sprintf('Stage 1 pixel MD    AUC=%.3f', s1.AUC), ...
        sprintf('CNN S2 regression   AUC=%.3f', cnnS2.AUC), ...
        sprintf('CNN latent MD       AUC=%.3f', cnnLat.AUC), ...
        sprintf('MLP S2 regression   AUC=%.3f', mlpS2.AUC), ...
        sprintf('MLP latent MD       AUC=%.3f', mlpLat.AUC), ...
        'Chance', ...
        'Location', 'southeast', 'FontSize', 9);
    title(sprintf('ROC Curves  |  normal: %s  vs  OOD: %s', normName, oodName), ...
        'FontWeight', 'bold');

    % -----------------------------------------------------------------------
    % Figure 2: Score distributions (normal vs OOD) for each score
    % -----------------------------------------------------------------------
    figure('Name', sprintf('Score Distributions: %s vs %s', normName, oodName), 'Color', 'w');
    tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    plotDist(nexttile, mdNorm.Stage1.MDScores, mdOod.Stage1.MDScores, ...
        'Stage 1: pixel MD', normName, oodName);

    nexttile;  axis off;   % placeholder for symmetric layout

    plotDist(nexttile, mdNorm.CNN.Stage2Scores(normS1), mdOod.CNN.Stage2Scores(oodS1), ...
        'CNN Stage 2 regression', normName, oodName);
    plotDist(nexttile, mdNorm.CNN.LatentScores(normS1), mdOod.CNN.LatentScores(oodS1), ...
        'CNN latent MD', normName, oodName);

    plotDist(nexttile, mdNorm.MLP.Stage2Scores(normS1), mdOod.MLP.Stage2Scores(oodS1), ...
        'MLP Stage 2 regression', normName, oodName);
    plotDist(nexttile, mdNorm.MLP.LatentScores(normS1), mdOod.MLP.LatentScores(oodS1), ...
        'MLP latent MD', normName, oodName);

    sgtitle(sprintf('Score Distributions  |  normal: %s  vs  OOD: %s', normName, oodName), ...
        'FontWeight', 'bold');

    % -----------------------------------------------------------------------
    % Pack results
    % -----------------------------------------------------------------------
    results = struct();
    results.NormalFolder = normalFolder;
    results.OODFolder    = oodFolder;
    results.ChexRoot     = chexRoot;
    results.Vigilance    = vigilance;
    results.Vigilance3   = vigilance3;

    results.Stage1      = s1;
    results.CNN.Stage2  = cnnS2;
    results.CNN.Latent  = cnnLat;
    results.MLP.Stage2  = mlpS2;
    results.MLP.Latent  = mlpLat;

    results.MDNormal = mdNorm;
    results.MDOod    = mdOod;
end

% ==========================================================================

function roc = buildROC(scores, labels)
% Compute ROC curve and AUC.  scores: higher = more in-distribution (normal).
% Negate so that higher score = more likely OOD = positive class (1).
    roc = struct('AUC', NaN, 'FPR', [], 'TPR', []);

    valid = ~isnan(scores);
    s = scores(valid);
    l = labels(valid);

    if numel(unique(l)) < 2
        warning('evaluate_auroc:singleClass', ...
            'Only one class present after NaN removal — AUC undefined.');
        return;
    end

    [roc.FPR, roc.TPR, ~, roc.AUC] = perfcurve(l, -s, 1);
end

function printRow(name, auc)
    if isnan(auc)
        fprintf('  %-32s  N/A\n', name);
    else
        fprintf('  %-32s  %.4f\n', name, auc);
    end
end

function plotROC(roc, lineSpec, label)
    if isnan(roc.AUC) || isempty(roc.FPR)
        return;
    end
    plot(roc.FPR, roc.TPR, lineSpec, 'LineWidth', 1.8, ...
        'DisplayName', sprintf('%s  AUC=%.3f', label, roc.AUC));
end

function plotDist(ax, normScores, oodScores, titleStr, normName, oodName)
    axes(ax);  %#ok<LAXES>
    hold on;
    histogram(normScores, 30, 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.6, ...
        'Normalization', 'probability');
    histogram(oodScores,  30, 'FaceColor', [0.8 0.3 0.2], 'FaceAlpha', 0.6, ...
        'Normalization', 'probability');
    hold off;
    xlabel('Score');  ylabel('Probability');
    legend(normName, oodName, 'Location', 'best', 'FontSize', 8);
    title(titleStr);
end
