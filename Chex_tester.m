function results = Chex_tester(testFolder, chexRoot, vigilance, forceRetrain, vigilance3)
% Chex_tester  Top-level driver for the CheXpert parallel OOD pipeline.
%
%   Calls MD_chex which runs:
%     Stage 1 (shared) – pixel-space MD prefilter
%     CNN track        – CNN regression + CNN latent-space MD (independent)
%     MLP track        – MLP regression + MLP latent-space MD (independent)
%
%   The two network tracks produce separate, non-combined verdicts.
%
%   RESULTS = Chex_tester()
%   RESULTS = Chex_tester(testFolder)
%   RESULTS = Chex_tester(testFolder, chexRoot)
%   RESULTS = Chex_tester(testFolder, chexRoot, vigilance)
%   RESULTS = Chex_tester(testFolder, chexRoot, vigilance, forceRetrain)
%   RESULTS = Chex_tester(testFolder, chexRoot, vigilance, forceRetrain, vigilance3)
%
%   Output struct fields:
%     .MDResults        full MD_chex output struct
%     .TestFolder       .ChexRoot   .Vigilance   .Vigilance3   .NumSamples
%     .Stage1Accepted   .Stage1Rejected
%     .CNNAccepted      .CNNRejected
%     .MLPAccepted      .MLPRejected

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
        vigilance3 = 0.5;
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
    % Run the pipeline
    % -----------------------------------------------------------------------
    fprintf('=== Running MD_chex (parallel pipeline) ===\n');
    mdResults = MD_chex(testFolder, chexRoot, vigilance, forceRetrain, vigilance3);

    numSamples     = mdResults.NumSamples;
    stage1Accepted = mdResults.Stage1.AcceptedCount;
    stage1Rejected = mdResults.Stage1.RejectedCount;
    cnnAccepted    = mdResults.CNN.AcceptedCount;
    cnnRejected    = mdResults.CNN.RejectedCount;
    mlpAccepted    = mdResults.MLP.AcceptedCount;
    mlpRejected    = mdResults.MLP.RejectedCount;

    fprintf('\n=== Chex_tester Summary ===\n');
    fprintf('  Total samples        : %d\n', numSamples);
    fprintf('  Stage 1 accepted     : %d  (%.1f%%)  — looks like a chest X-ray\n', ...
        stage1Accepted, 100 * stage1Accepted / numSamples);
    fprintf('  Stage 1 rejected     : %d  (%.1f%%)  — not a chest X-ray\n', ...
        stage1Rejected, 100 * stage1Rejected / numSamples);
    fprintf('  CNN track accepted   : %d  (%.1f%%)  — normal by CNN latent MD\n', ...
        cnnAccepted, 100 * cnnAccepted / numSamples);
    fprintf('  CNN track rejected   : %d  (%.1f%%)  — abnormal CNN latent features\n', ...
        cnnRejected, 100 * cnnRejected / numSamples);
    fprintf('  MLP track accepted   : %d  (%.1f%%)  — normal by MLP latent MD\n', ...
        mlpAccepted, 100 * mlpAccepted / numSamples);
    fprintf('  MLP track rejected   : %d  (%.1f%%)  — abnormal MLP latent features\n\n', ...
        mlpRejected, 100 * mlpRejected / numSamples);

    % -----------------------------------------------------------------------
    % Figure 1: Score distributions — 2x2 grid (CNN track top, MLP track bottom)
    % -----------------------------------------------------------------------
    s1Mask = mdResults.Stage1.Accepted;

    figure('Name', 'Chex_tester: Score Distributions', 'Color', 'w');
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % CNN Stage 2 regression
    nexttile;
    histogram(mdResults.CNN.Stage2Scores(s1Mask), 30, 'FaceColor', [0.2 0.5 0.8]);
    xlabel('Regression score');  ylabel('Count');
    title('CNN track — Stage 2 regression  (normal = 1.0)');

    % CNN latent MD
    nexttile;
    histogram(mdResults.CNN.LatentScores(s1Mask), 30, 'FaceColor', [0.2 0.5 0.8]);
    xline(vigilance3, 'r--', 'LineWidth', 1.5);
    xlabel('Latent MD confidence');  ylabel('Count');
    title(sprintf('CNN track — Stage 3 latent MD  (accepted=%d)', cnnAccepted));

    % MLP Stage 2 regression
    nexttile;
    histogram(mdResults.MLP.Stage2Scores(s1Mask), 30, 'FaceColor', [0.2 0.7 0.4]);
    xlabel('Regression score');  ylabel('Count');
    title('MLP track — Stage 2 regression  (normal = 1.0)');

    % MLP latent MD
    nexttile;
    histogram(mdResults.MLP.LatentScores(s1Mask), 30, 'FaceColor', [0.2 0.7 0.4]);
    xline(vigilance3, 'r--', 'LineWidth', 1.5);
    xlabel('Latent MD confidence');  ylabel('Count');
    title(sprintf('MLP track — Stage 3 latent MD  (accepted=%d)', mlpAccepted));

    sgtitle(sprintf( ...
        'S1 vig=%.2f  S3 vig=%.2f  |  S1 accepted=%d  |  CNN accepted=%d  MLP accepted=%d  (of %d)', ...
        vigilance, vigilance3, stage1Accepted, cnnAccepted, mlpAccepted, numSamples), ...
        'FontWeight', 'bold');

    % -----------------------------------------------------------------------
    % Figure 2: Stage-1 prefilter gallery
    % -----------------------------------------------------------------------
    showStage1Gallery(mdResults, vigilance, testFolder);

    % -----------------------------------------------------------------------
    % Figure 3: CNN track gallery
    % -----------------------------------------------------------------------
    showTrackGallery(mdResults, vigilance3, testFolder, 'CNN');

    % -----------------------------------------------------------------------
    % Figure 4: MLP track gallery
    % -----------------------------------------------------------------------
    showTrackGallery(mdResults, vigilance3, testFolder, 'MLP');

    % -----------------------------------------------------------------------
    % Output struct
    % -----------------------------------------------------------------------
    results = struct();
    results.MDResults      = mdResults;
    results.TestFolder     = testFolder;
    results.ChexRoot       = chexRoot;
    results.Vigilance      = vigilance;
    results.Vigilance3     = vigilance3;
    results.NumSamples     = numSamples;
    results.Stage1Accepted = stage1Accepted;
    results.Stage1Rejected = stage1Rejected;
    results.CNNAccepted    = cnnAccepted;
    results.CNNRejected    = cnnRejected;
    results.MLPAccepted    = mlpAccepted;
    results.MLPRejected    = mlpRejected;
end

% ==========================================================================

function showStage1Gallery(mdResults, vigilance, testFolder)
% Figure 2: 2x5 grid — Stage-1 prefilter accepted (top) / rejected (bottom).

    NUM_COLS = 5;

    acceptedIdx = find(mdResults.Stage1.Accepted);
    rejectedIdx = find(mdResults.Stage1.Rejected);

    numAcc = min(NUM_COLS, numel(acceptedIdx));
    numRej = min(NUM_COLS, numel(rejectedIdx));

    [~, testName] = fileparts(testFolder);

    fig = figure('Name', 'Chex_tester: Stage 1 Prefilter Gallery', 'Color', 'w');
    t   = tiledlayout(2, NUM_COLS, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('Stage 1 – Pixel-space MD  |  vigilance=%.2f  |  %s', vigilance, testName), ...
          'FontWeight', 'bold');

    for i = 1:NUM_COLS
        nexttile(i);
        if i <= numAcc
            idx = acceptedIdx(i);
            showStage1Tile(mdResults.TestFiles{idx}, 'ACCEPT', ...
                mdResults.Stage1.MDScores(idx), mdResults.Stage1.NormalizedDist(idx));
        else
            axis off;
        end
    end

    for i = 1:NUM_COLS
        nexttile(NUM_COLS + i);
        if i <= numRej
            idx = rejectedIdx(i);
            showStage1Tile(mdResults.TestFiles{idx}, 'REJECT', ...
                mdResults.Stage1.MDScores(idx), mdResults.Stage1.NormalizedDist(idx));
        else
            axis off;
        end
    end

    annotation(fig, 'textbox', [0 0.95 1 0.04], ...
        'String', sprintf('Accepted: %d    Rejected: %d    Source: %s', ...
            numel(acceptedIdx), numel(rejectedIdx), testFolder), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

function showStage1Tile(filepath, label, mdScore, normDist)
    img = imread(filepath);
    if size(img, 3) == 3,  img = rgb2gray(img);  end
    imshow(img, []);  axis image off;
    title({sprintf('%s  md=%.3f', label, mdScore), ...
           sprintf('dist=%.3f', normDist)}, 'FontSize', 9);
end

% ==========================================================================

function showTrackGallery(mdResults, vigilance3, testFolder, track)
% Figures 3 & 4: 2x5 grid for one network track.
%   Top row    – images accepted by S1 AND this track's latent filter.
%   Bottom row – images that passed S1 but were rejected by this track.

    NUM_COLS = 5;

    trackData   = mdResults.(track);
    acceptedIdx = find(trackData.Accepted);
    rejectedIdx = find(mdResults.Stage1.Accepted & ~trackData.Accepted);

    numAcc = min(NUM_COLS, numel(acceptedIdx));
    numRej = min(NUM_COLS, numel(rejectedIdx));

    [~, testName] = fileparts(testFolder);

    fig = figure('Name', sprintf('Chex_tester: %s Track Gallery', track), 'Color', 'w');
    t   = tiledlayout(2, NUM_COLS, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, sprintf('%s track – Stage 3 latent MD  |  vigilance=%.2f  |  %s', ...
          track, vigilance3, testName), 'FontWeight', 'bold');

    for i = 1:NUM_COLS
        nexttile(i);
        if i <= numAcc
            showTrackTile(mdResults, acceptedIdx(i), true, track);
        else
            axis off;
        end
    end

    for i = 1:NUM_COLS
        nexttile(NUM_COLS + i);
        if i <= numRej
            showTrackTile(mdResults, rejectedIdx(i), false, track);
        else
            axis off;
        end
    end

    annotation(fig, 'textbox', [0 0.95 1 0.04], ...
        'String', sprintf('Accepted: %d    Rejected: %d    Source: %s', ...
            numel(acceptedIdx), numel(rejectedIdx), testFolder), ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

function showTrackTile(mdResults, idx, isAccepted, track)
    img = imread(mdResults.TestFiles{idx});
    if size(img, 3) == 3,  img = rgb2gray(img);  end
    imshow(img, []);  axis image off;
    status = 'ACCEPT';
    if ~isAccepted,  status = 'REJECT';  end
    title({sprintf('%s  s2=%.3f', status, mdResults.(track).Stage2Scores(idx)), ...
           sprintf('latent=%.3f', mdResults.(track).LatentScores(idx))}, 'FontSize', 9);
end

% ==========================================================================

function forceRetrain = promptForceRetrain()
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
        forceRetrain = false;
        return;
    end

    fprintf('\nCached models found in %s\n', cacheDir);
    reply = input('  Use cached models? [Y/n]: ', 's');
    if isempty(reply),  reply = 'y';  end
    forceRetrain = strcmpi(reply, 'n');
end
