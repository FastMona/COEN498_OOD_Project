function results = mlp_custom_eval_metrics(normalFolder, oodFolder, trainFolder, W, b)
% mlp_custom_eval_metrics  AUROC, AUPR, and FPR@95TPR for the custom MLP.
%
%   Uses the last hidden layer (LHL) features of the hand-coded MLP
%   (testMLP / getLatentFeatures) with a pooled-covariance Mahalanobis
%   detector trained on the MNIST training split.
%
%   RESULTS = mlp_custom_eval_metrics(normalFolder, oodFolder, trainFolder, W, b)
%
%     normalFolder — folder with in-distribution IDX files (label=0)
%     oodFolder    — folder with OOD IDX files             (label=1)
%     trainFolder  — folder with MNIST training IDX files (for MD fitting)
%     W, b         — trained MLP weights from a completed training run
%
%   Metrics:
%     AUROC, AUPR-Out, AUPR-In, FPR@95TPR
%
%   Requires Statistics and Machine Learning Toolbox (perfcurve).

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
    if nargin < 4 || isempty(W) || isempty(b)
        error('mlp_custom_eval_metrics:noWeights', ...
            'W and b are required. Train the MLP first and pass the weights.');
    end

    if ~isfolder(normalFolder)
        error('mlp_custom_eval_metrics:badFolder', 'Normal folder not found: %s', normalFolder);
    end
    if ~isfolder(oodFolder)
        error('mlp_custom_eval_metrics:badFolder', 'OOD folder not found: %s', oodFolder);
    end

    [~, normName] = fileparts(normalFolder);
    [~, oodName]  = fileparts(oodFolder);

    fprintf('=== mlp_custom_eval_metrics ===\n');
    fprintf('  Normal  (label=0) : %s\n', normalFolder);
    fprintf('  OOD     (label=1) : %s\n', oodFolder);
    fprintf('  Training data     : %s\n', trainFolder);
    fprintf('\n');

    % -----------------------------------------------------------------------
    % Load images
    % -----------------------------------------------------------------------
    fprintf('Loading training data (for MD fit)...\n');
    [XTrain, YTrain] = loadMNISTForMLP(trainFolder);   % 784×N, labels 1–10

    fprintf('Loading normal images...\n');
    [XNorm, ~] = loadMNISTForMLP(normalFolder);

    fprintf('Loading OOD images...\n');
    [XOod, ~]  = loadMNISTForMLP(oodFolder);

    nN = size(XNorm, 2);
    nO = size(XOod,  2);
    fprintf('\n  Normal samples : %d    OOD samples : %d\n\n', nN, nO);

    labels = [zeros(nN, 1); ones(nO, 1)];

    % -----------------------------------------------------------------------
    % Extract last hidden layer features
    % -----------------------------------------------------------------------
    lhlIdx = numel(W) - 1;   % stop before output layer
    fprintf('Extracting LHL features (layer %d, %d-dim)...\n', lhlIdx, size(W{lhlIdx}, 1));

    FTrain = extractLHL(XTrain, W, b);   % N × d
    FNorm  = extractLHL(XNorm,  W, b);
    FOod   = extractLHL(XOod,   W, b);

    % -----------------------------------------------------------------------
    % Fit pooled-covariance Mahalanobis detector on training data
    % -----------------------------------------------------------------------
    fprintf('Fitting Mahalanobis detector...\n');
    mdStats = fitMD(FTrain, YTrain);

    % -----------------------------------------------------------------------
    % Score both folders (higher = more OOD)
    % -----------------------------------------------------------------------
    scNorm = minMahalanobis(FNorm, mdStats.classMeans, mdStats.invSigma);
    scOod  = minMahalanobis(FOod,  mdStats.classMeans, mdStats.invSigma);

    scores  = [scNorm; scOod];
    scoreLbl = 'Custom MLP  LHL (last hidden layer)';

    M = computeMetrics(scores, labels);

    % -----------------------------------------------------------------------
    % Print summary
    % -----------------------------------------------------------------------
    fprintf('\n=== Custom MLP Evaluation Metrics  |  normal: %s   OOD: %s ===\n', normName, oodName);
    fprintf('  %-36s  %6s  %8s  %7s  %9s\n', 'Score', 'AUROC', 'AUPR-Out', 'AUPR-In', 'FPR@95TPR');
    fprintf('  %s\n', repmat('-', 1, 75));
    fprintf('  %-36s  %6.4f  %8.4f  %7.4f  %8.2f%%\n', ...
        scoreLbl, M.AUROC, M.AUPR_Out, M.AUPR_In, 100*M.FPR95);
    fprintf('\n');

    % -----------------------------------------------------------------------
    % Figure 1: ROC
    % -----------------------------------------------------------------------
    figure('Name', sprintf('Custom MLP ROC: %s vs %s', normName, oodName), 'Color','w');
    if ~isnan(M.AUROC)
        plot(M.FPR, M.TPR, '-b', 'LineWidth', 1.8, ...
            'DisplayName', sprintf('%s  AUC=%.3f', scoreLbl, M.AUROC));
    end
    hold on;
    plot([0 1],[0 1],'k:','LineWidth',1,'HandleVisibility','off');
    hold off;
    xlabel('False Positive Rate');  ylabel('True Positive Rate');
    title(sprintf('Custom MLP ROC  |  normal: %s   OOD: %s', normName, oodName), 'FontWeight','bold');
    legend('Location','southeast','FontSize',8);
    grid on;  box on;

    % -----------------------------------------------------------------------
    % Figure 2: PR-Out
    % -----------------------------------------------------------------------
    figure('Name', sprintf('Custom MLP PR-Out: %s vs %s', normName, oodName), 'Color','w');
    if ~isnan(M.AUPR_Out)
        plot(M.RecOut, M.PrecOut, '-b', 'LineWidth', 1.8, ...
            'DisplayName', sprintf('%s  AUPR=%.3f', scoreLbl, M.AUPR_Out));
    end
    xlabel('Recall');  ylabel('Precision');
    title(sprintf('Custom MLP PR (OOD+)  |  normal: %s   OOD: %s', normName, oodName), 'FontWeight','bold');
    legend('Location','northeast','FontSize',8);
    grid on;  box on;

    % -----------------------------------------------------------------------
    % Figure 3: Score distribution
    % -----------------------------------------------------------------------
    figure('Name', sprintf('Custom MLP Score Distribution: %s vs %s', normName, oodName), 'Color','w');
    hold on;
    histogram(scNorm, 30, 'FaceColor',[0.2 0.5 0.8],'FaceAlpha',0.6,'Normalization','probability');
    histogram(scOod,  30, 'FaceColor',[0.8 0.3 0.2],'FaceAlpha',0.6,'Normalization','probability');
    hold off;
    xlabel('OOD Score (MD²)');  ylabel('Probability');
    legend(normName, oodName, 'Location','best','FontSize',8);
    title(scoreLbl);
    grid on;

    % -----------------------------------------------------------------------
    % Pack results
    % -----------------------------------------------------------------------
    results         = struct();
    results.NormalFolder = normalFolder;
    results.OODFolder    = oodFolder;
    results.TrainFolder  = trainFolder;
    results.MDStats      = mdStats;
    results.LHL          = M;
end

% =========================================================================
% FEATURE EXTRACTION
% =========================================================================

function F = extractLHL(X, W, b)
    % X: 784×N  →  F: N×d  (last hidden layer activations)
    h = X;
    for l = 1:numel(W)-1
        h = max(0, W{l} * h + b{l});
    end
    F = h';   % N × d
end

% =========================================================================
% MAHALANOBIS DETECTOR
% =========================================================================

function mdStats = fitMD(F, Y)
    % F: N×d features,  Y: N×1 labels (1–10)
    classes    = unique(Y);
    numClasses = numel(classes);
    d          = size(F, 2);
    N          = size(F, 1);

    classMeans = zeros(numClasses, d);
    for c = 1:numClasses
        classMeans(c,:) = mean(F(Y == classes(c), :), 1);
    end

    Sigma = zeros(d, d);
    for c = 1:numClasses
        Dc = F(Y == classes(c), :) - classMeans(c,:);
        Sigma = Sigma + Dc' * Dc;
    end
    Sigma   = Sigma / N;
    lambda  = 1e-3;
    invSigma = pinv(Sigma + lambda * eye(d));

    mdStats.classMeans = classMeans;
    mdStats.invSigma   = invSigma;
    mdStats.classes    = classes;
end

function scores = minMahalanobis(F, classMeans, invSigma)
    numClasses = size(classMeans, 1);
    dists = zeros(size(F,1), numClasses);
    for c = 1:numClasses
        D = F - classMeans(c,:);
        dists(:,c) = sum((D * invSigma) .* D, 2);
    end
    scores = min(dists, [], 2);
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
        warning('mlp_custom_eval_metrics:singleClass', 'Only one class present — metrics undefined.');
        return;
    end
    [m.FPR, m.TPR, ~, m.AUROC] = perfcurve(l, s, 1);
    idx = find(m.TPR >= 0.95, 1, 'first');
    if isempty(idx), m.FPR95 = 1.0; else, m.FPR95 = m.FPR(idx); end
    [m.RecOut, m.PrecOut, ~, m.AUPR_Out] = perfcurve(l,   s,  1, 'XCrit','reca','YCrit','prec');
    [m.RecIn,  m.PrecIn,  ~, m.AUPR_In]  = perfcurve(1-l, -s, 1, 'XCrit','reca','YCrit','prec');
end

% =========================================================================
% HELPERS
% =========================================================================

function folder = askFolder(defaultPath, label)
    fprintf('%s [%s]:\n', label, defaultPath);
    reply = strtrim(input('? ', 's'));
    if isempty(reply), folder = defaultPath; else, folder = reply; end
end
