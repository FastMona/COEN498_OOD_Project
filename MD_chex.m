function results = MD_chex(testInput, chexRoot, vigilance, forceRetrain, vigilance3)
% MD_chex  Three-stage OOD detector for CheXpert chest X-rays.
%
%   Orchestrates the full pipeline by delegating each MD stage:
%     Stage 1 – MD1_chex : pixel-space MD prefilter
%     Stage 2 – CNN_chex + MLP_chex regression scoring (diagnostic only)
%     Stage 3 – MD2_chex : latent-space MD filter
%
%   RESULTS = MD_chex(testInput)
%     testInput : path to a flat folder of 390x320 .jpg files
%                 OR an imageDatastore of pre-loaded images
%
%   RESULTS = MD_chex(testInput, chexRoot)
%     chexRoot  : path to the 390x320 training image folder (chex_train).
%
%   RESULTS = MD_chex(testInput, chexRoot, vigilance)
%     vigilance : Stage-1 confidence threshold in [0,1].  Default 0.5.
%
%   RESULTS = MD_chex(testInput, chexRoot, vigilance, forceRetrain)
%     forceRetrain : true ignores all caches and retrains from scratch.
%
%   RESULTS = MD_chex(testInput, chexRoot, vigilance, forceRetrain, vigilance3)
%     vigilance3 : Stage-3 confidence threshold in [0,1].  Default 0.7.
%                  Kept higher than vigilance — Stage 3 is the fine filter.
%
%   Output struct fields:
%     .Vigilance           – Stage-1 threshold
%     .Vigilance3          – Stage-3 threshold
%     .NumSamples          – total test images (N)
%     .TestFiles           – Nx1 cell of file paths
%
%     Stage 1 (from MD1_chex, all N images):
%     .PixelModel          – pixel-space PCA manifold struct
%     .MDScores            – Nx1 Stage-1 confidence
%     .NormalizedDist      – Nx1 normalised Mahalanobis distance
%     .Accepted            – Nx1 logical: passed Stage 1
%     .IsOOD               – Nx1 logical: rejected at Stage 1
%     .AcceptedCount       – number passed Stage 1
%     .RejectedCount       – number rejected at Stage 1
%
%     Stage 2 (NaN for Stage-1 rejects):
%     .CNN                 – CNN_chex results struct
%     .MLP                 – MLP_chex results struct
%     .CNNScores           – Nx1 CNN regression scores
%     .MLPScores           – Nx1 MLP regression scores
%     .CombinedScores      – Nx1 mean of CNN and MLP scores
%
%     Stage 3 (from MD2_chex, NaN / false for Stage-1 rejects):
%     .LatentModel         – struct with .CNN and .MLP latent manifolds
%     .CNNLatentScores     – Nx1 Stage-3 confidence from CNN latent space
%     .MLPLatentScores     – Nx1 Stage-3 confidence from MLP latent space
%     .LatentScores        – Nx1 mean latent confidence
%     .LatentAccepted      – Nx1 logical: passed both Stage 1 and Stage 3
%     .LatentAcceptedCount – number passing all stages
%     .LatentRejectedCount – number rejected at Stage 3

    if nargin < 1 || isempty(testInput)
        error('MD_chex:noInput', 'Provide a test folder path or imageDatastore.');
    end
    if nargin < 2 || isempty(chexRoot)
        here     = fileparts(mfilename('fullpath'));
        chexRoot = fullfile(here, 'chex_train');
    end
    if nargin < 3 || isempty(vigilance)
        vigilance = 0.5;
    end
    if nargin < 4
        forceRetrain = false;
    end
    if nargin < 5 || isempty(vigilance3)
        vigilance3 = 0.7;
    end
    if vigilance < 0 || vigilance > 1
        error('MD_chex:badVigilance',  'vigilance must be in [0, 1].');
    end
    if vigilance3 < 0 || vigilance3 > 1
        error('MD_chex:badVigilance3', 'vigilance3 must be in [0, 1].');
    end

    % -----------------------------------------------------------------------
    % Stage 1: pixel-space MD
    % -----------------------------------------------------------------------
    s1 = MD1_chex(testInput, chexRoot, vigilance, forceRetrain);

    N        = s1.NumSamples;
    accepted = s1.Accepted;

    % -----------------------------------------------------------------------
    % Stage 2 setup: load / train NN models
    % -----------------------------------------------------------------------
    fprintf('\nMD_chex: loading CNN model...\n');
    cnnResults = CNN_chex(chexRoot, forceRetrain);

    fprintf('MD_chex: loading MLP model...\n');
    mlpResults = MLP_chex(chexRoot, forceRetrain);

    cnnNet = cnnResults.Network;
    mlpNet = mlpResults.Network;

    % -----------------------------------------------------------------------
    % Allocate N-length output arrays (NaN / false for Stage-1 rejects)
    % -----------------------------------------------------------------------
    cnnScores      = nan(N, 1, 'single');
    mlpScores      = nan(N, 1, 'single');
    combinedScores = nan(N, 1, 'single');

    cnnLatentScores = nan(N, 1, 'single');
    mlpLatentScores = nan(N, 1, 'single');
    latentScores    = nan(N, 1, 'single');
    latentAccepted  = false(N, 1);
    latentModel     = [];

    latAcceptedCount = 0;
    latRejectedCount = 0;

    if s1.AcceptedCount > 0
        acceptedFiles = s1.TestFiles(accepted);
        acceptedImds  = imageDatastore(acceptedFiles, 'ReadFcn', @readAndPreprocess);

        % -------------------------------------------------------------------
        % Stage 2: NN regression scoring (diagnostic — not a gate)
        % The NNs are trained only on P (normal X-rays), so their regression
        % output for notP is undefined.  Scores are recorded for display
        % but do not filter images.
        % -------------------------------------------------------------------
        fprintf('\n--- Stage 2: NN regression ---\n');
        cnnAccepted  = predict(cnnNet, acceptedImds, 'MiniBatchSize', 32);
        mlpAccepted  = predict(mlpNet, acceptedImds, 'MiniBatchSize', 64);
        combAccepted = (cnnAccepted + mlpAccepted) / 2;

        cnnScores(accepted)      = cnnAccepted;
        mlpScores(accepted)      = mlpAccepted;
        combinedScores(accepted) = combAccepted;

        fprintf('  CNN  score: mean=%.4f  std=%.4f\n', mean(cnnAccepted),  std(cnnAccepted));
        fprintf('  MLP  score: mean=%.4f  std=%.4f\n', mean(mlpAccepted),  std(mlpAccepted));
        fprintf('  Combined  : mean=%.4f  std=%.4f\n', mean(combAccepted), std(combAccepted));
        fprintf('  (normal target = 1.0; lower = more abnormal)\n');

        % -------------------------------------------------------------------
        % Stage 3: latent-space MD via MD2_chex
        % -------------------------------------------------------------------
        s3 = MD2_chex(acceptedFiles, cnnNet, mlpNet, chexRoot, vigilance3, forceRetrain);

        % Map M-length Stage-3 results back into N-length arrays
        cnnLatentScores(accepted) = s3.CNNLatentScores;
        mlpLatentScores(accepted) = s3.MLPLatentScores;
        latentScores(accepted)    = s3.LatentScores;
        latentAccepted(accepted)  = s3.LatentAccepted;
        latentModel               = s3.LatentModel;
        latAcceptedCount          = s3.LatentAcceptedCount;
        latRejectedCount          = s3.LatentRejectedCount;
    else
        fprintf('  All samples rejected at Stage 1 — Stages 2 and 3 not run.\n');
    end

    % -----------------------------------------------------------------------
    % Pack combined results (preserves original MD_chex output contract)
    % -----------------------------------------------------------------------
    results = struct();
    results.Vigilance           = vigilance;
    results.Vigilance3          = vigilance3;
    results.NumSamples          = N;
    results.TestFiles           = s1.TestFiles;

    results.PixelModel          = s1.PixelModel;
    results.MDScores            = s1.MDScores;
    results.NormalizedDist      = s1.NormalizedDist;
    results.Accepted            = s1.Accepted;
    results.IsOOD               = s1.IsOOD;
    results.AcceptedCount       = s1.AcceptedCount;
    results.RejectedCount       = s1.RejectedCount;

    results.CNN                 = cnnResults;
    results.MLP                 = mlpResults;
    results.CNNScores           = cnnScores;
    results.MLPScores           = mlpScores;
    results.CombinedScores      = combinedScores;

    results.LatentModel         = latentModel;
    results.CNNLatentScores     = cnnLatentScores;
    results.MLPLatentScores     = mlpLatentScores;
    results.LatentScores        = latentScores;
    results.LatentAccepted      = latentAccepted;
    results.LatentAcceptedCount = latAcceptedCount;
    results.LatentRejectedCount = latRejectedCount;
end

% ==========================================================================
% Local helper — needed only for the Stage-2 acceptedImds datastore
% ==========================================================================

function img = readAndPreprocess(filename)
% Read a CheXpert JPEG, ensure single-channel [320x390x1] single in [0,1].
    img = imread(filename);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2single(img);
    if ismatrix(img)
        img = reshape(img, size(img, 1), size(img, 2), 1);
    end
end
