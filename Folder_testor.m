function results = Folder_testor(trainFolder, testFolder, rejectThreshold)
% Folder_testor Run manifold OOD rejection before CNN/MLP classification.
%
%   RESULTS = Folder_testor() prompts for training and testing folders,
%   then evaluates OOD samples using manifold filter followed by CNN/MLP.
%
%   RESULTS = Folder_testor(trainFolder, testFolder) uses specified folders.
%
%   RESULTS = Folder_testor(trainFolder, testFolder, rejectThreshold) uses
%   rejectThreshold as the MD_filter vigilance threshold in [0,1].

	% Get training folder from user
	if nargin < 1 || strlength(string(trainFolder)) == 0
		trainRoot = getSetFolderPaths('resolve', 'trainRoot');
	else
		trainRoot = getSetFolderPaths('resolve', 'trainRoot', trainFolder);
	end
	
	% Get testing folder from user
	if nargin < 2 || strlength(string(testFolder)) == 0
		oodFolder = getSetFolderPaths('resolve', 'testRoot');
	else
		oodFolder = getSetFolderPaths('resolve', 'testRoot', testFolder);
	end

	if nargin < 3 || isempty(rejectThreshold)
		rejectThreshold = 0.5;
	end
	if rejectThreshold < 0 || rejectThreshold > 1
		error('Folder_testor:badThreshold', ...
			'rejectThreshold must be between 0 and 1.');
	end
	fprintf('=== Folder Testor Configuration ===\n');
	fprintf('loading training data from: %s\n', getSetFolderPaths(trainRoot));
	fprintf('Running inference test on: %s\n', getSetFolderPaths(oodFolder));

	fprintf('\n=== OOD manifold filter ===\n');
	mdResults = MD_filter(oodFolder, rejectThreshold, false, false);
	fprintf('Vigilance: %.2f\n', rejectThreshold);
	fprintf('Samples:   %d\n', numel(mdResults.Accepted));
	fprintf('Accepted:  %d (%.2f%%)\n', mdResults.AcceptedCount, 100 * mdResults.AcceptedCount / numel(mdResults.Accepted));
	fprintf('Rejected:  %d (%.2f%%)\n', mdResults.RejectedCount, 100 * mdResults.RejectedCount / numel(mdResults.Accepted));

	cnnYPred = categorical.empty(0, 1);
	mlpYPred = categorical.empty(0, 1);
	agreementAccepted = NaN;
	distTbl = table(string(0:9)', zeros(10, 1), zeros(10, 1), ...
		'VariableNames', {'PredictedDigit', 'CNN_Count', 'MLP_Count'});

	if mdResults.AcceptedCount > 0
		fprintf('\n=== CNN / MLP on accepted samples only ===\n');
		cnnResults = CNN_reader(trainRoot);
		mlpResults = MLP_reader(trainRoot);

		acceptedMask = mdResults.Accepted;
		acceptedImages = mdResults.Images4D(:, :, :, acceptedMask);
		acceptedFeatures = mdResults.Features(acceptedMask, :);

		cnnYPred = classify(cnnResults.Network, acceptedImages, 'MiniBatchSize', 256);
		mlpYPred = classify(mlpResults.Network, acceptedFeatures, 'MiniBatchSize', 512);

		agreementAccepted = mean(cnnYPred == mlpYPred) * 100;
		fprintf('CNN vs MLP agreement: %.2f%%\n', agreementAccepted);

		predCats = categorical(string(0:9));
		digitNames = string(0:9)';
		cnnCounts = countcats(categorical(cellstr(string(cnnYPred)), cellstr(string(predCats))));
		mlpCounts = countcats(categorical(cellstr(string(mlpYPred)), cellstr(string(predCats))));
		distTbl = table(digitNames, cnnCounts(:), mlpCounts(:), ...
			'VariableNames', {'PredictedDigit', 'CNN_Count', 'MLP_Count'});

		disp('Accepted-sample digit distribution:');
		disp(distTbl);
	else
		fprintf('\n=== CNN / MLP on accepted samples only ===\n');
		fprintf('No samples passed the manifold OOD filter.\n');
	end

	results = struct();
	results.TrainDataRoot = trainRoot;
	results.OODDataRoot = oodFolder;
	results.RejectThreshold = rejectThreshold;
	results.MDFilter = mdResults;
	results.CNN = struct('YPred', cnnYPred);
	results.MLP = struct('YPred', mlpYPred);
	results.YOOD = mdResults.InputLabels;
	results.AgreementAcceptedPct = agreementAccepted;
	results.DigitDistribution = distTbl;
	results.AcceptedMask = mdResults.Accepted;
end
