function results = chex_eval_metrics(normalFolder, oodFolder, chexRoot)
% chex_eval_metrics  AUROC, AUPR, and FPR@95TPR for all pipeline scores.
%
%   Runs Chex_tester on normalFolder (in-distribution, label=0) and
%   oodFolder (OOD, label=1) with Stage 1 dormant so every image reaches
%   Stage 2 and Stage 3.  Computes four metrics for each of the nine
%   pipeline scores:
%
%     Stage 1         — pixel-space MD confidence
%     CNN  S2         — CNN regression score
%     CNN  S3 LHL     — last hidden layer MD
%     CNN  S3 FUSION  — multi-layer fused MD
%     CNN  S3 MBM     — multi-branch MD (max branch score)
%     MLP  S2         — MLP regression score
%     MLP  S3 LHL
%     MLP  S3 FUSION
%     MLP  S3 MBM
%
%   Metrics (per score):
%     AUROC     — Area Under the ROC curve (threshold-free)
%     AUPR-Out  — AUPR with OOD as positive class
%     AUPR-In   — AUPR with normal as positive class
%     FPR@95TPR — FPR when TPR = 95% (lower is better)
%
%   Requires Statistics and Machine Learning Toolbox (perfcurve).
%
%   RESULTS = chex_eval_metrics(normalFolder, oodFolder)
%   RESULTS = chex_eval_metrics(normalFolder, oodFolder, chexRoot)
%   Use Chex_cleanUp to clear model caches before a forced retrain.

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(normalFolder)
        normalFolder = askFolder(fullfile(here, 'chex_test'),      'Normal folder (label=0)');
    end
    if nargin < 2 || isempty(oodFolder)
        oodFolder    = askFolder(fullfile(here, 'chex_squares75'), 'OOD folder   (label=1)');
    end
    if nargin < 3 || isempty(chexRoot)
        chexRoot = fullfile(here, 'chex_train');
    end
    if ~isfolder(normalFolder)
        error('evaluation_metrics:badFolder', 'Normal folder not found: %s', normalFolder);
    end
    if ~isfolder(oodFolder)
        error('evaluation_metrics:badFolder', 'OOD folder not found: %s', oodFolder);
    end

    [~, normName] = fileparts(normalFolder);
    [~, oodName]  = fileparts(oodFolder);

    fprintf('=== evaluation_metrics ===\n');
    fprintf('  Normal  (label=0) : %s\n', normalFolder);
    fprintf('  OOD     (label=1) : %s\n', oodFolder);
    fprintf('  Training data     : %s\n', chexRoot);
    fprintf('\n');

    % -----------------------------------------------------------------------
    % Run Chex_tester on both folders.
    % rejectThreshold=0 and stage1Active=false → activeVigilance=0 → all
    % images pass Stage 1, so Stage 2 and Stage 3 scores are populated for
    % every image.  The raw Stage-1 MD scores are still recorded.
    % -----------------------------------------------------------------------
    fprintf('--- Scoring normal folder ---\n');
    rNorm = Chex_tester(chexRoot, normalFolder, 0, false);

    fprintf('\n--- Scoring OOD folder ---\n');
    rOod  = Chex_tester(chexRoot, oodFolder, 0, false);

    nN = rNorm.Stage1.NumSamples;
    nO = rOod.Stage1.NumSamples;
    fprintf('\n  Normal images : %d    OOD images : %d\n\n', nN, nO);

    % Labels: 0 = normal (ID), 1 = OOD
    labels = [zeros(nN, 1); ones(nO, 1)];

    % -----------------------------------------------------------------------
    % Collect scores.  Convention: higher score = more OOD.
    %
    %   Stage 1 / Stage 2 : confidence/regression → higher = more normal
    %                        → negate to get OOD score.
    %   Stage 3 LHL/FUSION : MD² → higher = more OOD → use directly.
    %   Stage 3 MBM        : N×B branch scores → normalise by per-branch
    %                        training threshold and take branch-wise max.
    % -----------------------------------------------------------------------
    scoreData = { ...
        negcat(rNorm.Stage1.MDScores,         rOod.Stage1.MDScores),         'Stage 1: pixel MD'; ...
        negcat(rNorm.CNN.Stage2Scores,         rOod.CNN.Stage2Scores),         'CNN  S2: regression'; ...
        poscat(rNorm.Stage3.CNN.LHL.Scores,    rOod.Stage3.CNN.LHL.Scores),    'CNN  S3 LHL'; ...
        poscat(rNorm.Stage3.CNN.FUSION.Scores, rOod.Stage3.CNN.FUSION.Scores), 'CNN  S3 FUSION'; ...
        mbmcat(rNorm.Stage3.CNN.MBM,           rOod.Stage3.CNN.MBM),           'CNN  S3 MBM'; ...
        negcat(rNorm.MLP.Stage2Scores,         rOod.MLP.Stage2Scores),         'MLP  S2: regression'; ...
        poscat(rNorm.Stage3.MLP.LHL.Scores,    rOod.Stage3.MLP.LHL.Scores),    'MLP  S3 LHL'; ...
        poscat(rNorm.Stage3.MLP.FUSION.Scores, rOod.Stage3.MLP.FUSION.Scores), 'MLP  S3 FUSION'; ...
        mbmcat(rNorm.Stage3.MLP.MBM,           rOod.Stage3.MLP.MBM),           'MLP  S3 MBM'};

    nScores = size(scoreData, 1);
    M = cell(nScores, 1);
    for k = 1:nScores
        M{k} = computeMetrics(scoreData{k,1}, labels);
    end

    % -----------------------------------------------------------------------
    % Print summary table
    % -----------------------------------------------------------------------
    fprintf('=== Evaluation Metrics  |  normal: %s   OOD: %s ===\n', normName, oodName);
    fprintf('  %-22s  %6s  %8s  %7s  %9s\n', ...
        'Score', 'AUROC', 'AUPR-Out', 'AUPR-In', 'FPR@95TPR');
    fprintf('  %s\n', repmat('-', 1, 60));
    for k = 1:nScores
        m = M{k};
        fprintf('  %-22s  %6.4f  %8.4f  %7.4f  %8.2f%%\n', ...
            scoreData{k,2}, m.AUROC, m.AUPR_Out, m.AUPR_In, 100 * m.FPR95);
    end
    fprintf('\n');

    % -----------------------------------------------------------------------
    % Figure 1: ROC curves
    % -----------------------------------------------------------------------
    colors = lines(nScores);
    lstyles = {'-', '--', '-', '--', ':', '-', '-', '--', ':'};

    figure('Name', sprintf('ROC: %s vs %s', normName, oodName), 'Color', 'w');
    hold on;
    for k = 1:nScores
        m = M{k};
        if ~isnan(m.AUROC)
            plot(m.FPR, m.TPR, lstyles{k}, 'Color', colors(k,:), 'LineWidth', 1.8, ...
                'DisplayName', sprintf('%s  AUC=%.3f', scoreData{k,2}, m.AUROC));
        end
    end
    plot([0 1], [0 1], 'k:', 'LineWidth', 1, 'HandleVisibility', 'off');
    hold off;
    xlabel('False Positive Rate');  ylabel('True Positive Rate');
    title(sprintf('ROC Curves  |  normal: %s   OOD: %s', normName, oodName), 'FontWeight', 'bold');
    legend('Location', 'southeast', 'FontSize', 8);
    grid on;  box on;

    % -----------------------------------------------------------------------
    % Figure 2: Precision-Recall curves (OOD as positive class)
    % -----------------------------------------------------------------------
    figure('Name', sprintf('PR-Out: %s vs %s', normName, oodName), 'Color', 'w');
    hold on;
    for k = 1:nScores
        m = M{k};
        if ~isnan(m.AUPR_Out)
            plot(m.RecOut, m.PrecOut, lstyles{k}, 'Color', colors(k,:), 'LineWidth', 1.8, ...
                'DisplayName', sprintf('%s  AUPR=%.3f', scoreData{k,2}, m.AUPR_Out));
        end
    end
    hold off;
    xlabel('Recall');  ylabel('Precision');
    title(sprintf('PR Curves (OOD positive)  |  normal: %s   OOD: %s', normName, oodName), ...
        'FontWeight', 'bold');
    legend('Location', 'northeast', 'FontSize', 8);
    grid on;  box on;

    % -----------------------------------------------------------------------
    % Figure 3: Score distributions (normal vs OOD)
    % -----------------------------------------------------------------------
    nRows = ceil(nScores / 2);
    figure('Name', sprintf('Score Distributions: %s vs %s', normName, oodName), 'Color', 'w');
    tl = tiledlayout(nRows, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    for k = 1:nScores
        sc    = scoreData{k,1};
        plotDist(nexttile, sc(1:nN), sc(nN+1:end), scoreData{k,2}, normName, oodName);
    end
    if mod(nScores, 2) == 1
        nexttile;  axis off;
    end
    title(tl, sprintf('Score Distributions  |  normal: %s   OOD: %s', normName, oodName), ...
        'FontWeight', 'bold');

    % -----------------------------------------------------------------------
    % Pack results
    % -----------------------------------------------------------------------
    results = struct();
    results.NormalFolder = normalFolder;
    results.OODFolder    = oodFolder;
    results.ChexRoot     = chexRoot;
    results.NormalRun    = rNorm;
    results.OODRun       = rOod;

    names = {'Stage1','CNN_S2','CNN_LHL','CNN_FUSION','CNN_MBM', ...
             'MLP_S2','MLP_LHL','MLP_FUSION','MLP_MBM'};
    for k = 1:nScores
        results.(names{k}) = M{k};
    end
end

% =========================================================================
% METRICS
% =========================================================================

function m = computeMetrics(scores, labels)
% Compute AUROC, AUPR-Out, AUPR-In, FPR@95TPR.
% scores: higher = more OOD.  labels: 0=normal, 1=OOD.
    m = struct('AUROC', NaN, 'AUPR_Out', NaN, 'AUPR_In', NaN, 'FPR95', NaN, ...
               'FPR', [], 'TPR', [], 'RecOut', [], 'PrecOut', [], ...
               'RecIn', [], 'PrecIn', []);

    s = double(scores(~isnan(scores)));
    l = double(labels(~isnan(scores)));

    if numel(unique(l)) < 2
        warning('evaluation_metrics:singleClass', ...
            'Only one class present after NaN removal — metrics undefined.');
        return;
    end

    % AUROC
    [m.FPR, m.TPR, ~, m.AUROC] = perfcurve(l, s, 1);

    % FPR at 95% TPR
    idx = find(m.TPR >= 0.95, 1, 'first');
    m.FPR95 = ternary(isempty(idx), 1.0, m.FPR(idx));

    % AUPR-Out: OOD (class=1) as positive
    [m.RecOut, m.PrecOut, ~, m.AUPR_Out] = ...
        perfcurve(l, s, 1, 'XCrit', 'reca', 'YCrit', 'prec');

    % AUPR-In: normal (class=0) as positive — flip labels and negate scores
    [m.RecIn, m.PrecIn, ~, m.AUPR_In] = ...
        perfcurve(1 - l, -s, 1, 'XCrit', 'reca', 'YCrit', 'prec');
end

% =========================================================================
% SCORE AGGREGATION HELPERS
% =========================================================================

function v = negcat(normSc, oodSc)
% Negate and concatenate: converts higher-is-normal to higher-is-OOD.
    v = [-double(normSc(:)); -double(oodSc(:))];
end

function v = poscat(normSc, oodSc)
% Concatenate without negation: MD² is already higher-is-OOD.
    v = [double(normSc(:)); double(oodSc(:))];
end

function v = mbmcat(normMBM, oodMBM)
% Aggregate N×B branch scores to a single OOD score per sample.
% Normalise each branch by its training threshold (so the decision boundary
% sits at 1.0 for every branch) then take the branch-wise maximum.
    thresh = max(double(normMBM.Threshold(:)'), 1e-6);   % 1×B
    normSc = max(double(normMBM.Scores) ./ thresh, [], 2);
    oodSc  = max(double(oodMBM.Scores)  ./ thresh, [], 2);
    v = [normSc; oodSc];
end

function v = ternary(cond, a, b)
    if cond, v = a; else, v = b; end
end

% =========================================================================
% PLOT HELPERS
% =========================================================================

function plotDist(ax, normSc, oodSc, titleStr, normName, oodName)
    axes(ax);  %#ok<LAXES>
    hold on;
    histogram(normSc, 30, 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.6, ...
        'Normalization', 'probability');
    histogram(oodSc,  30, 'FaceColor', [0.8 0.3 0.2], 'FaceAlpha', 0.6, ...
        'Normalization', 'probability');
    hold off;
    xlabel('OOD Score');  ylabel('Probability');
    legend(normName, oodName, 'Location', 'best', 'FontSize', 8);
    title(titleStr);
end

function folder = askFolder(defaultPath, label)
    fprintf('%s [%s]:\n', label, defaultPath);
    reply = strtrim(input('? ', 's'));
    if isempty(reply), folder = defaultPath; else, folder = reply; end
end
