function results = digit_eval_metrics(normalFolder, oodFolder, trainFolder)
% digit_eval_metrics  AUROC, AUPR, and FPR@95TPR for the MNIST digits pipeline.
%
%   Runs MD_Stage1_Prefilter (Stage 1 dormant) and MD_Stage3_Postfilter on
%   normalFolder (in-distribution, label=0) and oodFolder (OOD, label=1).
%   Computes four metrics for each of the seven pipeline scores:
%
%     Stage 1         — pixel-space MD confidence (best digit manifold)
%     CNN  S3 LHL     — last hidden layer MD
%     CNN  S3 FUSION  — multi-layer fused MD
%     CNN  S3 MBM     — multi-branch MD (max branch score)
%     MLP  S3 LHL
%     MLP  S3 FUSION
%     MLP  S3 MBM
%
%   Stage 2 (digit classification) does not produce a continuous OOD score
%   and is omitted from metric computation.
%
%   Metrics (per score):
%     AUROC     — Area Under the ROC curve (threshold-free)
%     AUPR-Out  — AUPR with OOD as positive class
%     AUPR-In   — AUPR with normal as positive class
%     FPR@95TPR — FPR when TPR = 95% (lower is better)
%
%   Requires Statistics and Machine Learning Toolbox (perfcurve).
%
%   RESULTS = digit_eval_metrics(normalFolder, oodFolder)
%   RESULTS = digit_eval_metrics(normalFolder, oodFolder, trainFolder)
%   Use cleanUp to clear model caches before a forced retrain.

    here = fileparts(mfilename('fullpath'));

    if nargin < 1 || isempty(normalFolder)
        normalFolder = askFolder(fullfile(here, 'MNIST_digits'), 'Normal folder (label=0)');
    end
    if nargin < 2 || isempty(oodFolder)
        oodFolder    = askFolder(fullfile(here, 'KMNIST_japanese'), 'OOD folder   (label=1)');
    end
    if nargin < 3 || isempty(trainFolder)
        trainFolder  = fullfile(here, 'MNIST_digits');
    end

    if ~isfolder(normalFolder)
        error('digit_eval_metrics:badFolder', 'Normal folder not found: %s', normalFolder);
    end
    if ~isfolder(oodFolder)
        error('digit_eval_metrics:badFolder', 'OOD folder not found: %s', oodFolder);
    end

    [~, normName] = fileparts(normalFolder);
    [~, oodName]  = fileparts(oodFolder);

    fprintf('=== digit_eval_metrics ===\n');
    fprintf('  Normal  (label=0) : %s\n', normalFolder);
    fprintf('  OOD     (label=1) : %s\n', oodFolder);
    fprintf('  Training data     : %s\n', trainFolder);
    fprintf('\n');

    % -----------------------------------------------------------------------
    % Stage 3 layer configurations  (must match Folder_testor)
    % CNN: conv1(8)→relu1→pool1 | conv2(16)→relu2→pool2 | conv3(32)→relu3 | fc1(64)→relu4
    % MLP: 784→512(relu1)→256(relu2)→128(relu3)→10
    % -----------------------------------------------------------------------
    cnnID  = 'CNN_8_16_32_fc64';
    cnnCfg = struct( ...
        'LHL',    'relu4', ...
        'FUSION', {{'relu1','relu2','relu3','relu4'}}, ...
        'MBM',    {{{'relu1'}, {'relu2'}, {'relu3','relu4'}}});

    mlpID  = 'MLP_512_256_128';
    mlpCfg = struct( ...
        'LHL',    'relu3', ...
        'FUSION', {{'relu1','relu2','relu3'}}, ...
        'MBM',    {{{'relu1'}, {'relu2'}, {'relu3'}}});

    algos = {'LHL', 'FUSION', 'MBM'};

    % -----------------------------------------------------------------------
    % Stage 1: score both folders (vigilance=0 → all pass, scores still computed)
    % -----------------------------------------------------------------------
    fprintf('--- Stage 1: scoring normal folder ---\n');
    s1Norm = MD_Stage1_Prefilter(normalFolder, 0, false, false);

    fprintf('--- Stage 1: scoring OOD folder ---\n');
    s1Ood  = MD_Stage1_Prefilter(oodFolder,    0, false, false);

    nN = numel(s1Norm.Confidence);
    nO = numel(s1Ood.Confidence);
    fprintf('\n  Normal samples : %d    OOD samples : %d\n\n', nN, nO);

    labels = [zeros(nN, 1); ones(nO, 1)];

    % -----------------------------------------------------------------------
    % Train CNN and MLP (loaded from cache when available)
    % -----------------------------------------------------------------------
    fprintf('--- Training / loading CNN and MLP ---\n');
    cnnRes = CNN_reader(trainFolder);
    mlpRes = MLP_reader(trainFolder);

    % -----------------------------------------------------------------------
    % Stage 3: train models then score both folders
    % -----------------------------------------------------------------------
    nets = { ...
        struct('tag','CNN', 'net',cnnRes.Network, 'id',cnnID, 'cfg',cnnCfg, ...
               'normX',s1Norm.Images4D,  'oodX',s1Ood.Images4D); ...
        struct('tag','MLP', 'net',mlpRes.Network, 'id',mlpID, 'cfg',mlpCfg, ...
               'normX',s1Norm.Features,  'oodX',s1Ood.Features)};

    stage3.CNN = struct('LHL',[], 'FUSION',[], 'MBM',[]);
    stage3.MLP = struct('LHL',[], 'FUSION',[], 'MBM',[]);

    for n = 1:numel(nets)
        nn = nets{n};
        for i = 1:numel(algos)
            algo = algos{i};
            lc   = nn.cfg.(algo);
            fprintf('  Training Stage3 %s %s...\n', nn.tag, algo);
            model  = MD_Stage3_Postfilter('train', nn.net, nn.id, algo, lc);
            s3Norm = MD_Stage3_Postfilter('test',  nn.net, nn.normX, model);
            s3Ood  = MD_Stage3_Postfilter('test',  nn.net, nn.oodX,  model);
            stage3.(nn.tag).(algo) = struct('Norm', s3Norm, 'OOD', s3Ood);
        end
    end

    % -----------------------------------------------------------------------
    % Collect scores — convention: higher = more OOD
    % Stage 1 Confidence: higher = more normal → negate
    % Stage 3 LHL/FUSION Scores: MD² higher = more OOD → use directly
    % Stage 3 MBM Scores: N×B → normalise by threshold, take branch-wise max
    % -----------------------------------------------------------------------
    scoreData = { ...
        negcat(s1Norm.Confidence,                      s1Ood.Confidence),                      'Stage 1: pixel MD'; ...
        poscat(stage3.CNN.LHL.Norm.Scores,             stage3.CNN.LHL.OOD.Scores),             'CNN  S3 LHL'; ...
        poscat(stage3.CNN.FUSION.Norm.Scores,          stage3.CNN.FUSION.OOD.Scores),          'CNN  S3 FUSION'; ...
        mbmcat(stage3.CNN.MBM.Norm,                    stage3.CNN.MBM.OOD),                    'CNN  S3 MBM'; ...
        poscat(stage3.MLP.LHL.Norm.Scores,             stage3.MLP.LHL.OOD.Scores),             'MLP  S3 LHL'; ...
        poscat(stage3.MLP.FUSION.Norm.Scores,          stage3.MLP.FUSION.OOD.Scores),          'MLP  S3 FUSION'; ...
        mbmcat(stage3.MLP.MBM.Norm,                    stage3.MLP.MBM.OOD),                    'MLP  S3 MBM'};

    nScores = size(scoreData, 1);
    M = cell(nScores, 1);
    for k = 1:nScores
        M{k} = computeMetrics(scoreData{k,1}, labels);
    end

    % -----------------------------------------------------------------------
    % Print summary table
    % -----------------------------------------------------------------------
    fprintf('\n=== Evaluation Metrics  |  normal: %s   OOD: %s ===\n', normName, oodName);
    fprintf('  %-22s  %6s  %8s  %7s  %9s\n', 'Score', 'AUROC', 'AUPR-Out', 'AUPR-In', 'FPR@95TPR');
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
    colors  = lines(nScores);
    lstyles = {'-', '-', '--', ':', '-', '--', ':'};

    figure('Name', sprintf('ROC: %s vs %s', normName, oodName), 'Color', 'w');
    hold on;
    for k = 1:nScores
        m = M{k};
        if ~isnan(m.AUROC)
            plot(m.FPR, m.TPR, lstyles{k}, 'Color', colors(k,:), 'LineWidth', 1.8, ...
                'DisplayName', sprintf('%s  AUC=%.3f', scoreData{k,2}, m.AUROC));
        end
    end
    plot([0 1],[0 1],'k:','LineWidth',1,'HandleVisibility','off');
    hold off;
    xlabel('False Positive Rate');  ylabel('True Positive Rate');
    title(sprintf('ROC Curves  |  normal: %s   OOD: %s', normName, oodName), 'FontWeight','bold');
    legend('Location','southeast','FontSize',8);
    grid on;  box on;

    % -----------------------------------------------------------------------
    % Figure 2: PR curves (OOD as positive)
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
        'FontWeight','bold');
    legend('Location','northeast','FontSize',8);
    grid on;  box on;

    % -----------------------------------------------------------------------
    % Figure 3: Score distributions
    % -----------------------------------------------------------------------
    nRows = ceil(nScores / 2);
    figure('Name', sprintf('Score Distributions: %s vs %s', normName, oodName), 'Color','w');
    tl = tiledlayout(nRows, 2, 'TileSpacing','compact','Padding','compact');
    for k = 1:nScores
        sc = scoreData{k,1};
        plotDist(nexttile, sc(1:nN), sc(nN+1:end), scoreData{k,2}, normName, oodName);
    end
    if mod(nScores,2)==1, nexttile; axis off; end
    title(tl, sprintf('Score Distributions  |  normal: %s   OOD: %s', normName, oodName), ...
        'FontWeight','bold');

    % -----------------------------------------------------------------------
    % Pack results
    % -----------------------------------------------------------------------
    results = struct();
    results.NormalFolder = normalFolder;
    results.OODFolder    = oodFolder;
    results.TrainFolder  = trainFolder;
    results.NormalS1     = s1Norm;
    results.OODS1        = s1Ood;
    results.Stage3       = stage3;

    names = {'Stage1','CNN_LHL','CNN_FUSION','CNN_MBM','MLP_LHL','MLP_FUSION','MLP_MBM'};
    for k = 1:nScores
        results.(names{k}) = M{k};
    end
end

% =========================================================================
% METRICS
% =========================================================================

function m = computeMetrics(scores, labels)
    m = struct('AUROC',NaN,'AUPR_Out',NaN,'AUPR_In',NaN,'FPR95',NaN, ...
               'FPR',[],'TPR',[],'RecOut',[],'PrecOut',[],'RecIn',[],'PrecIn',[]);
    s = double(scores(~isnan(scores)));
    l = double(labels(~isnan(scores)));
    if numel(unique(l)) < 2
        warning('digit_eval_metrics:singleClass', 'Only one class present — metrics undefined.');
        return;
    end
    [m.FPR, m.TPR, ~, m.AUROC] = perfcurve(l, s, 1);
    idx = find(m.TPR >= 0.95, 1, 'first');
    m.FPR95 = ternary(isempty(idx), 1.0, m.FPR(idx));
    [m.RecOut, m.PrecOut, ~, m.AUPR_Out] = perfcurve(l,   s,  1, 'XCrit','reca','YCrit','prec');
    [m.RecIn,  m.PrecIn,  ~, m.AUPR_In]  = perfcurve(1-l, -s, 1, 'XCrit','reca','YCrit','prec');
end

% =========================================================================
% SCORE AGGREGATION
% =========================================================================

function v = negcat(normSc, oodSc)
    v = [-double(normSc(:)); -double(oodSc(:))];
end

function v = poscat(normSc, oodSc)
    v = [double(normSc(:)); double(oodSc(:))];
end

function v = mbmcat(normMBM, oodMBM)
    thresh = max(double(normMBM.Threshold(:)'), 1e-6);
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
    histogram(normSc, 30, 'FaceColor',[0.2 0.5 0.8],'FaceAlpha',0.6,'Normalization','probability');
    histogram(oodSc,  30, 'FaceColor',[0.8 0.3 0.2],'FaceAlpha',0.6,'Normalization','probability');
    hold off;
    xlabel('OOD Score');  ylabel('Probability');
    legend(normName, oodName, 'Location','best','FontSize',8);
    title(titleStr);
end

function folder = askFolder(defaultPath, label)
    fprintf('%s [%s]:\n', label, defaultPath);
    reply = strtrim(input('? ', 's'));
    if isempty(reply), folder = defaultPath; else, folder = reply; end
end
