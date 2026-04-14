function results = Chex_tester(testFolder, chexRoot, vigilance, forceRetrain, vigilance3)
% Chex_tester  Top-level driver for the three-stage CheXpert OOD pipeline.
%
%   Calls MD_chex which runs:
%     Stage 1 – pixel-space MD prefilter  ("Is this a chest X-ray?")
%     Stage 2 – CNN_chex + MLP_chex regression scoring
%     Stage 3 – latent-space MD filter    ("Does this X-ray look normal?")
%
%   RESULTS = Chex_tester()
%     Trains (or loads) all models on .\chex_train, then scores .\chex_train
%     as a sanity check (all-normal baseline).
%
%   RESULTS = Chex_tester(testFolder)
%     Scores images in testFolder (e.g. CheXpert_pacemaker or random OOD).
%
%   RESULTS = Chex_tester(testFolder, chexRoot)
%     chexRoot : training image folder (default: .\chex_train).
%
%   RESULTS = Chex_tester(testFolder, chexRoot, vigilance)
%     vigilance : MD acceptance threshold applied at Stage 1 and Stage 3.
%                 Default 0.5.
%
%   RESULTS = Chex_tester(testFolder, chexRoot, vigilance, forceRetrain)
%     forceRetrain : true forces retraining of all models and manifolds.
%
%   Output struct fields:
%     .MDResults            – full MD_chex output struct (all stage details)
%     .TestFolder           – path that was scored
%     .ChexRoot             – training folder used
%     .Vigilance            – threshold applied
%     .NumSamples           – total images scored
%     .Stage1Accepted       – passed pixel-space MD (looks like an X-ray)
%     .Stage1Rejected       – rejected at Stage 1
%     .Stage3Accepted       – passed both Stage 1 and latent-space MD
%     .Stage3Rejected       – passed Stage 1 but failed Stage 3 (abnormal)

    if nargin < 1 || isempty(testFolder)
        testFolder = getSetFolderPaths('resolve', 'testRoot');
    else
        testFolder = getSetFolderPaths('resolve', 'testRoot', testFolder);
    end
    if nargin < 2 || isempty(chexRoot)
        chexRoot = getSetFolderPaths('resolve', 'trainRoot');
    else
        chexRoot = getSetFolderPaths('resolve', 'trainRoot', chexRoot);
    end
    if nargin < 3 || isempty(vigilance)
        vigilance = 0.5;
    end
    if nargin < 4
        forceRetrain = promptForceRetrain();
    end
    if nargin < 5 || isempty(vigilance3)
        vigilance3 = 0.7;
    end

    if vigilance < 0 || vigilance > 1
        error('Chex_tester:badVigilance', 'vigilance must be in [0, 1].');
    end
    if ~isfolder(testFolder)
        error('Chex_tester:missingTestFolder', 'Test folder not found: %s', testFolder);
    end
    if ~isfolder(chexRoot)
        error('Chex_tester:missingChexRoot', ...
            'Training folder not found: %s\nRun pad_chex.py first.', chexRoot);
    end

    fprintf('=== Chex_tester Configuration ===\n');
    fprintf('  Training data : %s\n', chexRoot);
    fprintf('  Test data     : %s\n', testFolder);
    fprintf('  Vigilance S1  : %.2f\n', vigilance);
    fprintf('  Vigilance S3  : %.2f\n\n', vigilance3);

    % -----------------------------------------------------------------------
    % Run the full three-stage pipeline
    % -----------------------------------------------------------------------
    fprintf('=== Running MD_chex (3-stage pipeline) ===\n');
    mdResults = MD_chex(testFolder, chexRoot, vigilance, forceRetrain, vigilance3);

    numSamples      = mdResults.NumSamples;
    stage1Accepted  = mdResults.AcceptedCount;
    stage1Rejected  = mdResults.RejectedCount;
    stage3Accepted  = mdResults.LatentAcceptedCount;
    stage3Rejected  = mdResults.LatentRejectedCount;

    fprintf('\n=== Chex_tester Summary ===\n');
    fprintf('  Total samples        : %d\n', numSamples);
    fprintf('  Stage 1 accepted     : %d  (%.1f%%)  — looks like a chest X-ray\n', ...
        stage1Accepted, 100 * stage1Accepted / numSamples);
    fprintf('  Stage 1 rejected     : %d  (%.1f%%)  — not a chest X-ray\n', ...
        stage1Rejected, 100 * stage1Rejected / numSamples);
    fprintf('  Stage 3 accepted     : %d  (%.1f%%)  — normal X-ray features\n', ...
        stage3Accepted, 100 * stage3Accepted / numSamples);
    fprintf('  Stage 3 rejected     : %d  (%.1f%%)  — abnormal latent features\n\n', ...
        stage3Rejected, 100 * stage3Rejected / numSamples);

    % -----------------------------------------------------------------------
    % Plots
    % -----------------------------------------------------------------------
    figure('Name', 'Chex_tester: Pipeline Score Distributions', 'Color', 'w');
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Stage 1: pixel MD confidence
    nexttile;
    histogram(mdResults.MDScores, 40, 'FaceColor', [0.4 0.4 0.8]);
    xline(vigilance, 'r--', 'LineWidth', 1.5);
    xlabel('Confidence');  ylabel('Count');
    title(sprintf('Stage 1: pixel MD  (accepted=%d)', stage1Accepted));

    % Stage 2: NN regression scores (accepted only)
    accepted1 = mdResults.Accepted;
    nexttile;
    hold on;
    histogram(mdResults.CNNScores(accepted1),      30, 'FaceColor', [0.2 0.5 0.8], 'FaceAlpha', 0.6);
    histogram(mdResults.MLPScores(accepted1),      30, 'FaceColor', [0.2 0.7 0.4], 'FaceAlpha', 0.6);
    histogram(mdResults.CombinedScores(accepted1), 30, 'FaceColor', [0.8 0.4 0.2], 'FaceAlpha', 0.6);
    hold off;
    legend('CNN', 'MLP', 'Combined', 'Location', 'northwest');
    xlabel('Regression score');  ylabel('Count');
    title('Stage 2: NN scores  (normal target = 1.0)');

    % Stage 3: CNN latent MD confidence
    nexttile;
    histogram(mdResults.CNNLatentScores(accepted1), 30, 'FaceColor', [0.2 0.5 0.8]);
    xline(vigilance3, 'r--', 'LineWidth', 1.5);
    xlabel('Confidence');  ylabel('Count');
    title(sprintf('Stage 3: CNN latent MD  (accepted=%d)', stage3Accepted));

    % Stage 3: MLP latent MD confidence
    nexttile;
    histogram(mdResults.MLPLatentScores(accepted1), 30, 'FaceColor', [0.2 0.7 0.4]);
    xline(vigilance3, 'r--', 'LineWidth', 1.5);
    xlabel('Confidence');  ylabel('Count');
    title('Stage 3: MLP latent MD');

    sgtitle(sprintf('S1 vigilance=%.2f  S3 vigilance=%.2f  |  S1 accepted=%d  S3 accepted=%d  (of %d)', ...
        vigilance, vigilance3, stage1Accepted, stage3Accepted, numSamples), 'FontWeight', 'bold');

    % -----------------------------------------------------------------------
    % Sample gallery: 2 rows x 5 cols  (accepted top, rejected bottom)
    % -----------------------------------------------------------------------
    showGallery(mdResults, vigilance, testFolder);

    % -----------------------------------------------------------------------
    % Output struct
    % -----------------------------------------------------------------------
    results = struct();
    results.MDResults       = mdResults;
    results.TestFolder      = testFolder;
    results.ChexRoot        = chexRoot;
    results.Vigilance       = vigilance;
    results.NumSamples      = numSamples;
    results.Stage1Accepted  = stage1Accepted;
    results.Stage1Rejected  = stage1Rejected;
    results.Stage3Accepted  = stage3Accepted;
    results.Stage3Rejected  = stage3Rejected;
end

% ==========================================================================

function showGallery(mdResults, vigilance, testFolder)
% showGallery  Display a 2×5 grid of representative accepted / rejected images.
%
%   Top row    – Stage-3 accepted (passed both Stage 1 and Stage 3).
%   Bottom row – Stage-3 rejected (passed Stage 1, failed Stage 3).
%                Falls back to Stage-1 rejects when too few Stage-3 rejects exist.
%
%   Up to 5 images are shown per row; empty tiles are blanked.

    NUM_COLS = 5;

    acceptedIdx = find(mdResults.LatentAccepted);

    % Prefer Stage-3 rejects; pad with Stage-1 rejects if needed.
    stage3RejIdx = find(mdResults.Accepted & ~mdResults.LatentAccepted);
    stage1RejIdx = find(~mdResults.Accepted);
    rejectedIdx  = [stage3RejIdx; stage1RejIdx];

    numAcc = min(NUM_COLS, numel(acceptedIdx));
    numRej = min(NUM_COLS, numel(rejectedIdx));

    [~, testName] = fileparts(testFolder);

    figure('Name', 'Chex_tester: Sample Gallery', 'Color', 'w');
    t   = tiledlayout(2, NUM_COLS, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Sample Gallery  |  vigilance=%.2f  |  %s', vigilance, testName), ...
          'FontWeight', 'bold');

    % --- Top row: accepted ---
    for i = 1:NUM_COLS
        nexttile(i);
        if i <= numAcc
            idx   = acceptedIdx(i);
            score = mdResults.LatentScores(idx);
            showGalleryTile(mdResults.TestFiles{idx}, 'ACCEPT', score);
        else
            axis off;
        end
    end

    % --- Bottom row: rejected ---
    for i = 1:NUM_COLS
        nexttile(NUM_COLS + i);
        if i <= numRej
            idx = rejectedIdx(i);
            if ~mdResults.Accepted(idx)
                label = 'S1-REJ';
                score = mdResults.MDScores(idx);
            else
                label = 'S3-REJ';
                score = mdResults.LatentScores(idx);
            end
            showGalleryTile(mdResults.TestFiles{idx}, label, score);
        else
            axis off;
        end
    end
end

function showGalleryTile(filepath, label, score)
% Load one image, display it, and add a concise title.
    img = imread(filepath);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    imshow(img, []);
    axis image off;
    title(sprintf('%s  %.3f', label, score), 'FontSize', 9);
end

% ==========================================================================

function forceRetrain = promptForceRetrain()
% If any trained_models cache exists, ask the user whether to reuse it.
    here     = fileparts(mfilename('fullpath'));
    cacheDir = fullfile(here, 'trained_models');
    caches   = { 'cnn_chex_cache.mat', 'mlp_chex_cache.mat', ...
                 'md_chex_cache.mat',  'md_chex_latent_cache.mat' };

    anyFound = false;
    for i = 1:numel(caches)
        if isfile(fullfile(cacheDir, caches{i}))
            anyFound = true;
            break;
        end
    end

    if ~anyFound
        % Nothing cached yet — training is unavoidable, no need to ask.
        forceRetrain = false;
        return;
    end

    fprintf('\nCached models found in %s\n', cacheDir);
    reply = input('  Use cached models? [Y/n]: ', 's');
    if isempty(reply)
        reply = 'y';
    end
    forceRetrain = strcmpi(reply, 'n');
end
