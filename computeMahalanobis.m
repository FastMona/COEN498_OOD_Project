function [scores, mdStats] = computeMahalanobis(ZTrain, YTrain, ZQuery, mdStats)
% computeMahalanobis  Compute Mahalanobis distance-based OOD scores.
%
%   [scores, mdStats] = computeMahalanobis(ZTrain, YTrain, ZQuery)
%   [scores, ~]       = computeMahalanobis([], [], ZQuery, mdStats)
%
%   Implements the Mahalanobis distance OOD detector from
%   Lee et al. (NeurIPS 2018), with the shared covariance formulation
%   used in Venkataramanan et al. (ICCVW 2023).
%
%   The OOD score for a query sample z is:
%
%       DM(z) = min_c  (z - mu_c)' * Sigma^{-1} * (z - mu_c)
%
%   where mu_c is the centroid of class c and Sigma is the shared (tied)
%   covariance matrix estimated from all training samples (Eq. 1 in
%   Venkataramanan et al., 2023).  A small ridge (1e-6 * I) is added to
%   Sigma before inversion to avoid singularities, as recommended by
%   Ren et al. (2021).
%
%   High scores indicate OOD; low scores indicate in-distribution.
%
%   Inputs:
%       ZTrain  - (N x d) matrix of TRAINING latent features
%       YTrain  - (N x 1) categorical labels for training samples
%       ZQuery  - (M x d) matrix of query (test/OOD) latent features
%       mdStats - (optional) pre-computed stats struct from a prior call
%                 (pass to avoid recomputing on same training set)
%
%   Outputs:
%       scores  - (M x 1) OOD scores (minimum Mahalanobis distance)
%       mdStats - struct with fields:
%           .classMeans   (nClasses x d) per-class centroid matrix
%           .SigmaInv     (d x d)        inverse shared covariance
%           .classLabels  cell array of class label strings
%           .nComp        latent feature dimensionality d

    % ------------------------------------------------------------------
    % 1. Compute (or reuse) class statistics
    % ------------------------------------------------------------------
    if nargin < 4 || isempty(mdStats)
        mdStats = fitMahalanobisStats(ZTrain, YTrain);
    end

    % ------------------------------------------------------------------
    % 2. Compute per-class Mahalanobis distances for every query sample
    % ------------------------------------------------------------------
    nClasses = size(mdStats.classMeans, 1);
    nQuery   = size(ZQuery, 1);
    Dmc      = zeros(nQuery, nClasses);   % D_Mc for each class c

    SigInv = mdStats.SigmaInv;

    for c = 1 : nClasses
        diff       = ZQuery - mdStats.classMeans(c, :);   % M x d
        % Squared Mahalanobis: diag(diff * SigInv * diff')
        % Efficient vectorised form: sum((diff * SigInv) .* diff, 2)
        Dmc(:, c)  = sum((diff * SigInv) .* diff, 2);
    end

    % OOD score = minimum squared MD to any class centroid (Eq. 3 in
    % Anthony & Kamnitsas 2023 / Lee et al. 2018)
    scores = min(Dmc, [], 2);
end


% =========================================================================
% Helper: fit class statistics from training data
% =========================================================================
function mdStats = fitMahalanobisStats(ZTrain, YTrain)

    ZTrain = double(ZTrain);
    cats   = categories(YTrain);
    nCls   = numel(cats);
    d      = size(ZTrain, 2);

    fprintf('[computeMahalanobis] Fitting class statistics (%d classes, d=%d)...\n', ...
            nCls, d);

    classMeans = zeros(nCls, d);
    Sigma      = zeros(d, d);
    N          = size(ZTrain, 1);

    for c = 1 : nCls
        mask          = YTrain == cats{c};
        Zc            = ZTrain(mask, :);
        classMeans(c,:) = mean(Zc, 1);

        % Accumulate unnormalised scatter matrix (tied/shared covariance)
        diff   = Zc - classMeans(c, :);
        Sigma  = Sigma + diff' * diff;
    end

    % Normalise by N (not N-1) – consistent with Lee et al. 2018
    Sigma = Sigma / N;

    % Ridge regularisation to avoid singular matrix (Ren et al. 2021)
    ridge = 1e-6 * eye(d);
    SigmaReg = Sigma + ridge;

    mdStats.classMeans  = classMeans;
    mdStats.SigmaInv    = inv(SigmaReg);
    mdStats.classLabels = cats;
    mdStats.nComp       = d;

    fprintf('[computeMahalanobis] Stats computed. Ridge = 1e-6.\n');
end