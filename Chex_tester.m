function results = Chex_tester(testFolder, chexRoot, rejectThreshold, forceRetrain)
% Chex_tester  Top-level driver: train CheXpert models and score a test set.
%
%   RESULTS = Chex_tester()
%     Trains (or loads) CNN_chex and MLP_chex on .\chex_train, then scores
%     the same chex_train folder as a sanity check (all-normal baseline).
%
%   RESULTS = Chex_tester(testFolder)
%     Scores the images in testFolder after ensuring models are trained.
%     Pass a folder of CheXpert_pacemaker (or other OOD) images here.
%
%   RESULTS = Chex_tester(testFolder, chexRoot)
%     chexRoot : training image folder (default: .\chex_train).
%
%   RESULTS = Chex_tester(testFolder, chexRoot, rejectThreshold)
%     rejectThreshold : combined score below this value is flagged as OOD.
%                       Default: 0.5.
%
%   RESULTS = Chex_tester(testFolder, chexRoot, rejectThreshold, forceRetrain)
%     forceRetrain : true forces CNN and MLP retraining.
%
%   Output struct fields:
%     .MDResults       – full MD_chex output struct
%     .TestFolder      – path that was scored
%     .ChexRoot        – training folder used
%     .RejectThreshold – threshold applied
%     .NumSamples      – total images scored
%     .NumAccepted     – images with CombinedScore >= rejectThreshold
%     .NumRejected     – images with CombinedScore <  rejectThreshold
%     .AcceptedPct     – accepted percentage
%     .RejectedPct     – rejected percentage

    % Resolve folders through getSetFolderPaths (prompts when not provided)
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
    if nargin < 3 || isempty(rejectThreshold)
        rejectThreshold = 0.9;
    end
    if nargin < 4
        forceRetrain = false;
    end

    if rejectThreshold < 0 || rejectThreshold > 1
        error('Chex_tester:badThreshold', 'rejectThreshold must be in [0, 1].');
    end
    if ~isfolder(testFolder)
        error('Chex_tester:missingTestFolder', ...
            'Test folder not found: %s', testFolder);
    end
    if ~isfolder(chexRoot)
        error('Chex_tester:missingChexRoot', ...
            'Training folder not found: %s\nRun pad_chex.py first.', chexRoot);
    end

    fprintf('=== Chex_tester Configuration ===\n');
    fprintf('  Training data : %s\n', chexRoot);
    fprintf('  Test data     : %s\n', testFolder);
    fprintf('  Threshold     : %.2f\n\n', rejectThreshold);

    % -----------------------------------------------------------------------
    % Run MD_chex (trains models if needed, scores test folder)
    % -----------------------------------------------------------------------
    fprintf('=== Running MD_chex ===\n');
    mdResults = MD_chex(testFolder, chexRoot, forceRetrain);

    % -----------------------------------------------------------------------
    % Apply threshold
    % -----------------------------------------------------------------------
    accepted = mdResults.CombinedScores >= rejectThreshold;
    numSamples  = mdResults.NumSamples;
    numAccepted = sum(accepted);
    numRejected = numSamples - numAccepted;

    fprintf('\n=== Chex_tester Results ===\n');
    fprintf('  Samples    : %d\n',          numSamples);
    fprintf('  Threshold  : %.2f\n',         rejectThreshold);
    fprintf('  Accepted   : %d  (%.1f%%)\n', numAccepted, 100 * numAccepted / numSamples);
    fprintf('  Rejected   : %d  (%.1f%%)\n', numRejected, 100 * numRejected / numSamples);
    fprintf('  CNN  score : mean=%.4f  std=%.4f\n', ...
        mean(mdResults.CNNScores), std(mdResults.CNNScores));
    fprintf('  MLP  score : mean=%.4f  std=%.4f\n', ...
        mean(mdResults.MLPScores), std(mdResults.MLPScores));
    fprintf('  Combined   : mean=%.4f  std=%.4f\n\n', ...
        mean(mdResults.CombinedScores), std(mdResults.CombinedScores));

    % -----------------------------------------------------------------------
    % Score distribution plot
    % -----------------------------------------------------------------------
    figure('Name', 'CheXpert Anomaly Score Distribution', 'Color', 'w');

    tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    histogram(mdResults.CNNScores, 40, 'FaceColor', [0.2 0.5 0.8]);
    xline(rejectThreshold, 'r--', 'LineWidth', 1.5);
    xlabel('Score');  ylabel('Count');
    title(sprintf('CNN  (mean=%.3f)', mdResults.MeanCNN));

    nexttile;
    histogram(mdResults.MLPScores, 40, 'FaceColor', [0.2 0.7 0.4]);
    xline(rejectThreshold, 'r--', 'LineWidth', 1.5);
    xlabel('Score');
    title(sprintf('MLP  (mean=%.3f)', mdResults.MeanMLP));

    nexttile;
    histogram(mdResults.CombinedScores, 40, 'FaceColor', [0.8 0.4 0.2]);
    xline(rejectThreshold, 'r--', 'LineWidth', 1.5);
    xlabel('Score');
    title(sprintf('Combined  (mean=%.3f)', mdResults.MeanCombined));

    sgtitle(sprintf('Anomaly scores  |  accepted=%d  rejected=%d  threshold=%.2f', ...
        numAccepted, numRejected, rejectThreshold), 'FontWeight', 'bold');

    % -----------------------------------------------------------------------
    % Output struct
    % -----------------------------------------------------------------------
    results = struct();
    results.MDResults       = mdResults;
    results.TestFolder      = testFolder;
    results.ChexRoot        = chexRoot;
    results.RejectThreshold = rejectThreshold;
    results.NumSamples      = numSamples;
    results.NumAccepted     = numAccepted;
    results.NumRejected     = numRejected;
    results.AcceptedPct     = 100 * numAccepted / numSamples;
    results.RejectedPct     = 100 * numRejected / numSamples;
end
