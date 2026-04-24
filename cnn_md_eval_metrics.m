function results = cnn_md_eval_metrics(normalFolder, oodFolder)
% cnn_md_eval_metrics  AUROC, AUPR, and FPR@95TPR for the CNN_MD legacy models.
%
%   Evaluates two CNN_MD models trained by the Final_a / Final_b scripts:
%     CNN-MD-a  LHL    — single layer fc1 (mnist_cnn_model_with_md.mat)
%     CNN-MD-b  FUSION — relu1, relu2, relu3, fc1 concatenated
%                        (mnist_cnn_model_with_multilayer_md.mat)
%
%   Both models use a shared pooled-covariance Mahalanobis detector.
%   The OOD score is the minimum class-conditional MD (higher = more OOD).
%
%   Metrics (per score):
%     AUROC     — Area Under the ROC curve
%     AUPR-Out  — AUPR with OOD as positive class
%     AUPR-In   — AUPR with normal as positive class
%     FPR@95TPR — FPR when TPR = 95%
%
%   RESULTS = cnn_md_eval_metrics(normalFolder, oodFolder)
%   Run CNN_MD_Train_Digit_Final_a and _b first to produce the model files.

    here        = fileparts(mfilename('fullpath'));
    modelFolder = fullfile(here, 'trained_models');

    if nargin < 1 || isempty(normalFolder)
        normalFolder = askFolder(fullfile(here, 'MNIST_digits'), 'Normal folder (label=0)');
    end
    if nargin < 2 || isempty(oodFolder)
        oodFolder = askFolder(fullfile(here, 'KMNIST_japanese'), 'OOD folder   (label=1)');
    end

    if ~isfolder(normalFolder)
        error('cnn_md_eval_metrics:badFolder', 'Normal folder not found: %s', normalFolder);
    end
    if ~isfolder(oodFolder)
        error('cnn_md_eval_metrics:badFolder', 'OOD folder not found: %s', oodFolder);
    end

    modelA = fullfile(modelFolder, 'mnist_cnn_model_with_md.mat');
    modelB = fullfile(modelFolder, 'mnist_cnn_model_with_multilayer_md.mat');

    if ~isfile(modelA)
        error('cnn_md_eval_metrics:missingModel', ...
            'LHL model not found: %s\nRun CNN_MD_Train_Digit_Final_a first.', modelA);
    end
    if ~isfile(modelB)
        error('cnn_md_eval_metrics:missingModel', ...
            'FUSION model not found: %s\nRun CNN_MD_Train_Digit_Final_b first.', modelB);
    end

    [~, normName] = fileparts(normalFolder);
    [~, oodName]  = fileparts(oodFolder);

    fprintf('=== cnn_md_eval_metrics ===\n');
    fprintf('  Normal  (label=0) : %s\n', normalFolder);
    fprintf('  OOD     (label=1) : %s\n', oodFolder);
    fprintf('\n');

    % -----------------------------------------------------------------------
    % Load models
    % -----------------------------------------------------------------------
    fprintf('Loading CNN_MD_a (LHL)...\n');
    Sa = load(modelA, 'net', 'mdStats');
    fprintf('Loading CNN_MD_b (FUSION)...\n');
    Sb = load(modelB, 'net', 'mdStats');

    % -----------------------------------------------------------------------
    % Load images from both folders
    % -----------------------------------------------------------------------
    fprintf('Loading normal images...\n');
    XNorm = loadImages(normalFolder);
    fprintf('Loading OOD images...\n');
    XOod  = loadImages(oodFolder);

    nN = size(XNorm, 4);
    nO = size(XOod,  4);
    fprintf('\n  Normal samples : %d    OOD samples : %d\n\n', nN, nO);

    labels = [zeros(nN, 1); ones(nO, 1)];

    % -----------------------------------------------------------------------
    % Score with each model
    % -----------------------------------------------------------------------
    fprintf('Scoring with CNN-MD-a  LHL (%s)...\n', Sa.mdStats.featureLayer);
    scA_norm = scoreLHL(Sa.net, XNorm, Sa.mdStats);
    scA_ood  = scoreLHL(Sa.net, XOod,  Sa.mdStats);

    fprintf('Scoring with CNN-MD-b  FUSION (%s)...\n', ...
        strjoin(Sb.mdStats.featureLayers, ', '));
    scB_norm = scoreFusion(Sb.net, XNorm, Sb.mdStats);
    scB_ood  = scoreFusion(Sb.net, XOod,  Sb.mdStats);

    % -----------------------------------------------------------------------
    % Collect scores — MD higher = more OOD, use directly
    % -----------------------------------------------------------------------
    scoreData = { ...
        [scA_norm; scA_ood], 'CNN-MD-a  LHL    (fc1)'; ...
        [scB_norm; scB_ood], 'CNN-MD-b  FUSION (relu1,relu2,relu3,fc1)'};

    nScores = size(scoreData, 1);
    M = cell(nScores, 1);
    for k = 1:nScores
        M{k} = computeMetrics(scoreData{k,1}, labels);
    end

    % -----------------------------------------------------------------------
    % Print summary table
    % -----------------------------------------------------------------------
    fprintf('\n=== CNN-MD Evaluation Metrics  |  normal: %s   OOD: %s ===\n', normName, oodName);
    fprintf('  %-42s  %6s  %8s  %7s  %9s\n', 'Score', 'AUROC', 'AUPR-Out', 'AUPR-In', 'FPR@95TPR');
    fprintf('  %s\n', repmat('-', 1, 80));
    for k = 1:nScores
        m = M{k};
        fprintf('  %-42s  %6.4f  %8.4f  %7.4f  %8.2f%%\n', ...
            scoreData{k,2}, m.AUROC, m.AUPR_Out, m.AUPR_In, 100*m.FPR95);
    end
    fprintf('\n');

    % -----------------------------------------------------------------------
    % Figure 1: ROC curves
    % -----------------------------------------------------------------------
    colors = lines(nScores);

    figure('Name', sprintf('CNN-MD ROC: %s vs %s', normName, oodName), 'Color','w');
    hold on;
    for k = 1:nScores
        m = M{k};
        if ~isnan(m.AUROC)
            plot(m.FPR, m.TPR, '-', 'Color', colors(k,:), 'LineWidth', 1.8, ...
                'DisplayName', sprintf('%s  AUC=%.3f', scoreData{k,2}, m.AUROC));
        end
    end
    plot([0 1],[0 1],'k:','LineWidth',1,'HandleVisibility','off');
    hold off;
    xlabel('False Positive Rate');  ylabel('True Positive Rate');
    title(sprintf('CNN-MD ROC Curves  |  normal: %s   OOD: %s', normName, oodName), 'FontWeight','bold');
    legend('Location','southeast','FontSize',8);
    grid on;  box on;

    % -----------------------------------------------------------------------
    % Figure 2: PR curves (OOD as positive)
    % -----------------------------------------------------------------------
    figure('Name', sprintf('CNN-MD PR-Out: %s vs %s', normName, oodName), 'Color','w');
    hold on;
    for k = 1:nScores
        m = M{k};
        if ~isnan(m.AUPR_Out)
            plot(m.RecOut, m.PrecOut, '-', 'Color', colors(k,:), 'LineWidth', 1.8, ...
                'DisplayName', sprintf('%s  AUPR=%.3f', scoreData{k,2}, m.AUPR_Out));
        end
    end
    hold off;
    xlabel('Recall');  ylabel('Precision');
    title(sprintf('CNN-MD PR Curves (OOD+)  |  normal: %s   OOD: %s', normName, oodName), ...
        'FontWeight','bold');
    legend('Location','northeast','FontSize',8);
    grid on;  box on;

    % -----------------------------------------------------------------------
    % Figure 3: Score distributions
    % -----------------------------------------------------------------------
    figure('Name', sprintf('CNN-MD Score Distributions: %s vs %s', normName, oodName), 'Color','w');
    tl = tiledlayout(1, 2, 'TileSpacing','compact','Padding','compact');
    for k = 1:nScores
        sc = scoreData{k,1};
        plotDist(nexttile, sc(1:nN), sc(nN+1:end), scoreData{k,2}, normName, oodName);
    end
    title(tl, sprintf('CNN-MD Score Distributions  |  normal: %s   OOD: %s', normName, oodName), ...
        'FontWeight','bold');

    % -----------------------------------------------------------------------
    % Pack results
    % -----------------------------------------------------------------------
    results = struct();
    results.NormalFolder = normalFolder;
    results.OODFolder    = oodFolder;
    results.LHL          = M{1};
    results.FUSION       = M{2};
end

% =========================================================================
% SCORING
% =========================================================================

function scores = scoreLHL(net, X, mdStats)
    F = activations(net, X, mdStats.featureLayer, 'OutputAs', 'rows');
    scores = minMahalanobis(double(F), mdStats.classMeans, mdStats.invSigma);
end

function scores = scoreFusion(net, X, mdStats)
    F = extractFusionFeatures(net, X, mdStats.featureLayers, mdStats.fusionStats);
    scores = minMahalanobis(F, mdStats.classMeans, mdStats.invSigma);
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

function F = extractFusionFeatures(net, X, layerNames, fusionStats)
    blocks = cell(numel(layerNames), 1);
    for k = 1:numel(layerNames)
        A = activations(net, X, layerNames{k});
        if ndims(A) == 4
            A = squeeze(mean(mean(A, 1), 2));
            if isvector(A), A = A(:)'; else, A = A'; end
        else
            A = activations(net, X, layerNames{k}, 'OutputAs', 'rows');
        end
        A = double(A);
        mu    = fusionStats(k).mu;
        sigma = fusionStats(k).sigma;
        sigma(sigma < 1e-8) = 1;
        blocks{k} = (A - mu) ./ sigma;
    end
    F = cat(2, blocks{:});
end

% =========================================================================
% IMAGE LOADING
% =========================================================================

function X = loadImages(folder)
    candidates = {'t10k-images-idx3-ubyte', 'train-images-idx3-ubyte'};
    imgFile = '';
    for i = 1:numel(candidates)
        f = fullfile(folder, candidates{i});
        if isfile(f)
            imgFile = f;
            break;
        end
    end
    if isempty(imgFile)
        error('cnn_md_eval_metrics:noImages', 'No IDX image file found in: %s', folder);
    end
    X = readIDXImages(imgFile);
end

function images4D = readIDXImages(filePath)
    fid = fopen(filePath, 'rb');
    if fid < 0, error('Cannot open: %s', filePath); end
    cleaner = onCleanup(@() fclose(fid));
    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magic ~= 2051
        error('Invalid IDX magic number in %s (got %d)', filePath, magic);
    end
    numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
    numRows   = fread(fid, 1, 'int32', 0, 'ieee-be');
    numCols   = fread(fid, 1, 'int32', 0, 'ieee-be');
    pixels    = fread(fid, numImages * numRows * numCols, 'uint8=>single');
    images3D  = reshape(pixels, [numCols, numRows, numImages]);
    images3D  = permute(images3D, [2, 1, 3]) ./ 255;
    images4D  = reshape(images3D, [numRows, numCols, 1, numImages]);
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
        warning('cnn_md_eval_metrics:singleClass', 'Only one class present — metrics undefined.');
        return;
    end
    [m.FPR, m.TPR, ~, m.AUROC] = perfcurve(l, s, 1);
    idx = find(m.TPR >= 0.95, 1, 'first');
    if isempty(idx), m.FPR95 = 1.0; else, m.FPR95 = m.FPR(idx); end
    [m.RecOut, m.PrecOut, ~, m.AUPR_Out] = perfcurve(l,   s,  1, 'XCrit','reca','YCrit','prec');
    [m.RecIn,  m.PrecIn,  ~, m.AUPR_In]  = perfcurve(1-l, -s, 1, 'XCrit','reca','YCrit','prec');
end

% =========================================================================
% PLOT HELPERS
% =========================================================================

function plotDist(ax, normSc, oodSc, titleStr, normName, oodName)
    axes(ax); %#ok<LAXES>
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
