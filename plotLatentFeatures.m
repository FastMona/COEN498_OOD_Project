function plotLatentFeatures(ZId, YId, ZOod, YOod, titleStr)
% plotLatentFeatures  Visualise ID and OOD samples in the latent space.
%
%   Produces a 1x2 figure:
%     Left  - first two PCA dimensions (fast, exact)
%     Right - 2-D t-SNE embedding      (richer non-linear structure)
%
%   OOD samples are plotted with black 'x' markers.

    if nargin < 5, titleStr = 'Latent Space'; end

    % ------------------------------------------------------------------
    % 0. Subsample for speed
    % ------------------------------------------------------------------
    MAX_ID  = 3000;
    MAX_OOD = 1000;

    if size(ZId, 1) > MAX_ID
        idx = randperm(size(ZId, 1), MAX_ID);
        ZId = ZId(idx, :);
        YId = YId(idx);
    end
    if size(ZOod, 1) > MAX_OOD
        idx  = randperm(size(ZOod, 1), MAX_OOD);
        ZOod = ZOod(idx, :);
        if ~isempty(YOod), YOod = YOod(idx); end
    end

    nId  = size(ZId,  1);
    nOod = size(ZOod, 1);

    % ------------------------------------------------------------------
    % 1. Build colour index over ID points only (length = nId)
    % ------------------------------------------------------------------
    cats   = categories(YId);
    nCls   = numel(cats);
    cmap   = lines(nCls);

    colIdx = zeros(nId, 1);
    for c = 1:nCls
        colIdx(YId == cats{c}) = c;
    end

    % ------------------------------------------------------------------
    % 2. Concatenate for joint embedding; track which rows are OOD
    % ------------------------------------------------------------------
    ZAll  = [ZId; ZOod];                          % (nId+nOod) x d
    isOod = [false(nId, 1); true(nOod, 1)];       % same length as ZAll

    % ------------------------------------------------------------------
    % 3. PCA projection to 2-D
    % ------------------------------------------------------------------
    if size(ZAll, 2) > 2
        [~, Zpc] = pca(double(ZAll), 'NumComponents', 2);
    else
        Zpc = double(ZAll);
    end

    % ------------------------------------------------------------------
    % 4. t-SNE embedding
    % ------------------------------------------------------------------
    fprintf('[plotLatentFeatures] Running t-SNE (n=%d)...\n', size(ZAll,1));
    nPCA_tsne = min(50, size(ZAll, 2));
    Ztsne = tsne(double(ZAll), ...
                 'NumDimensions',    2, ...
                 'Perplexity',       30, ...
                 'NumPCAComponents', nPCA_tsne, ...
                 'Algorithm',        'barneshut');

    % ------------------------------------------------------------------
    % 5. Plot
    % ------------------------------------------------------------------
    figure('Name', titleStr, 'Color', 'w', 'Position', [100 100 1200 520]);

    viewNames = {'PCA (PC1 vs PC2)', 't-SNE'};
    projs     = {Zpc, Ztsne};

    for v = 1:2
        ax = subplot(1, 2, v);
        hold(ax, 'on');
        Z2 = projs{v};

        % ID samples – iterate over classes using ID indices only
        for c = 1:nCls
            % colIdx and ~isOod(1:nId) are both length nId
            idxId = find(~isOod);              % indices of ID rows in ZAll
            mask  = colIdx == c;               % length nId
            pts   = idxId(mask);               % rows in ZAll for this class
            if ~isempty(pts)
                scatter(ax, Z2(pts,1), Z2(pts,2), 12, ...
                        cmap(c,:), 'filled', 'MarkerFaceAlpha', 0.6);
            end
        end

        % OOD samples – black crosses
        idxOod = find(isOod);
        if ~isempty(idxOod)
            scatter(ax, Z2(idxOod,1), Z2(idxOod,2), 20, ...
                    'k', 'x', 'LineWidth', 1.2);
        end

        % Legend
        legEntries = arrayfun(@(c) sprintf('ID: %s', cats{c}), ...
                              (1:nCls)', 'UniformOutput', false);
        if ~isempty(idxOod)
            if ~isempty(YOod)
                oodCats  = unique(YOod);
                oodLabel = sprintf('OOD (digits: %s)', ...
                    strjoin(arrayfun(@char, oodCats, 'UniformOutput', false), ', '));
            else
                oodLabel = 'OOD';
            end
            legEntries{end+1} = oodLabel; %#ok<AGROW>
        end

        legend(ax, legEntries, 'Location', 'bestoutside', 'FontSize', 7);
        title(ax, sprintf('%s – %s', titleStr, viewNames{v}));
        xlabel(ax, 'Dim 1');
        ylabel(ax, 'Dim 2');
        box(ax, 'on');
        grid(ax, 'on');
    end

    sgtitle(titleStr, 'FontWeight', 'bold', 'FontSize', 13);
    fprintf('[plotLatentFeatures] Figure rendered.\n');
end