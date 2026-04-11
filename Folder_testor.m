function results = Folder_testor(oodFolder, rejectThreshold)
% Folder_testor Run manifold OOD rejection before CNN/MLP classification.
%
%   RESULTS = Folder_testor() first applies MD_filter to KMNIST_japanese using
%   vigilance 0.7, then sends only accepted samples to the cached CNN and MLP
%   classifiers trained on MNIST_digits/raw.
%
%   RESULTS = Folder_testor(oodFolder) evaluates OOD samples from oodFolder.
%
%   RESULTS = Folder_testor(oodFolder, rejectThreshold) uses rejectThreshold
%   as the MD_filter vigilance threshold in [0,1].

	if nargin < 1 || strlength(string(oodFolder)) == 0
		here = fileparts(mfilename('fullpath'));
		oodFolder = fullfile(here, 'KMNIST_japanese');
	end

	if nargin < 2 || isempty(rejectThreshold)
		rejectThreshold = 0.7;
	end
	if rejectThreshold < 0 || rejectThreshold > 1
		error('Folder_testor:badThreshold', ...
			'rejectThreshold must be between 0 and 1.');
	end

	here = fileparts(mfilename('fullpath'));
	trainRoot = fullfile(here, 'MNIST_digits', 'raw');

	fprintf('=== OOD manifold filter ===\n');
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
