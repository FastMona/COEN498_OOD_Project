function [Z, pcaModel] = getLatentFeatures(net, X, pcaModel, opts)
% getLatentFeatures  Extract and optionally PCA-reduce latent features.
%
%   [Z, pcaModel] = getLatentFeatures(net, X)
%   [Z, pcaModel] = getLatentFeatures(net, X, [], opts)   % fit new PCA
%   [Z, ~]        = getLatentFeatures(net, X, pcaModel)   % apply existing
%
%   Extracts activations from the 'relu_latent' layer of the trained MLP,
%   then applies PCA dimensionality reduction.  Venkataramanan et al.
%   (ICCVW 2023) show that PCA on latent features improves Mahalanobis
%   distance-based OOD detection by up to 30% by removing non-informative
%   dimensions that dilute the covariance structure.  The number of
%   components is chosen to retain >= varThresh of the total variance.
%
%   Inputs:
%       net      - trained network (output of trainMLP)
%       X        - (N x 784) single matrix of images
%       pcaModel - (optional) struct from a previous call; if supplied the
%                  same projection is re-used (for test/OOD data).
%                  Pass [] or omit to fit a new PCA on X (training data).
%       opts     - (optional) struct with fields:
%           .layerName   name of layer to tap (default: 'relu_latent')
%           .varThresh   PCA variance threshold in (0,1] (default: 0.95)
%           .miniBatch   batch size for activations (default: 512)
%           .applyPCA    toggle PCA on/off (default: true)
%
%   Outputs:
%       Z        - (N x d') matrix of (PCA-reduced) latent features
%       pcaModel - struct with fields .coeff, .mu, .nComp, .explainedSum
%                  (empty if applyPCA == false)

    % ------------------------------------------------------------------
    % 0. Defaults
    % ------------------------------------------------------------------
    if nargin < 3, pcaModel = []; end
    if nargin < 4, opts = struct(); end

    layerName  = getOpt(opts, 'layerName',  'relu_latent');
    varThresh  = getOpt(opts, 'varThresh',  0.95);
    miniBatch  = getOpt(opts, 'miniBatch',  512);
    applyPCA   = getOpt(opts, 'applyPCA',   true);

    % ------------------------------------------------------------------
    % 1. Extract raw activations in mini-batches
    % ------------------------------------------------------------------
    N   = size(X, 1);
    Z_raw = [];

    fprintf('[getLatentFeatures] Extracting activations from "%s"...\n', layerName);
    for i = 1 : ceil(N / miniBatch)
        batchIdx = (i-1)*miniBatch+1 : min(i*miniBatch, N);
        acts = activations(net, X(batchIdx, :), layerName, ...
                           'MiniBatchSize', miniBatch, ...
                           'OutputAs', 'rows');
        Z_raw = [Z_raw; acts]; %#ok<AGROW>
    end
    fprintf('[getLatentFeatures] Raw feature dim: %d  (N=%d)\n', size(Z_raw,2), N);

    % ------------------------------------------------------------------
    % 2. PCA dimensionality reduction
    % ------------------------------------------------------------------
    if ~applyPCA
        Z        = Z_raw;
        pcaModel = [];
        return
    end

    if isempty(pcaModel)
        % --- Fit PCA on training features ---
        fprintf('[getLatentFeatures] Fitting PCA (variance threshold=%.0f%%)...\n', ...
                varThresh*100);

        [coeff, score, ~, ~, explained, mu] = pca(double(Z_raw));

        % Choose number of components to explain >= varThresh variance
        cumExp = cumsum(explained) / 100;
        nComp  = find(cumExp >= varThresh, 1, 'first');
        if isempty(nComp), nComp = size(coeff, 2); end

        pcaModel.coeff        = coeff(:, 1:nComp);
        pcaModel.mu           = mu;
        pcaModel.nComp        = nComp;
        pcaModel.explainedSum = cumExp(nComp) * 100;

        Z = score(:, 1:nComp);
        fprintf('[getLatentFeatures] PCA: kept %d / %d components (%.1f%% var)\n', ...
                nComp, size(coeff,2), pcaModel.explainedSum);
    else
        % --- Apply existing PCA transform (test / OOD data) ---
        Z_centered = double(Z_raw) - pcaModel.mu;
        Z = Z_centered * pcaModel.coeff;
        fprintf('[getLatentFeatures] Applied existing PCA (%d components)\n', ...
                pcaModel.nComp);
    end
end


% -------------------------------------------------------------------------
function v = getOpt(s, field, default)
    if isfield(s, field)
        v = s.(field);
    else
        v = default;
    end
end