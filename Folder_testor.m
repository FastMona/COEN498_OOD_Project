function results = Folder_testor(trainFolder, testFolder, rejectThreshold, stage1Active)
% Folder_testor  Three-stage OOD pipeline: pre-filter, classify, post-filter.
%
%   RESULTS = Folder_testor()
%     Resolves paths interactively and runs the full pipeline.
%
%   RESULTS = Folder_testor(trainFolder, testFolder)
%   RESULTS = Folder_testor(trainFolder, testFolder, rejectThreshold)
%   RESULTS = Folder_testor(trainFolder, testFolder, rejectThreshold, stage1Active)
%     rejectThreshold  vigilance for Stage 1 MD_Stage1_Prefilter [0,1] default 0.5
%     stage1Active     logical, default false — set true to enable Stage 1 filtering.
%                      When false Stage 1 runs with vigilance=0 (all samples pass).
%
%   Pipeline
%     Stage 1 : MD_Stage1_Prefilter  — pixel-space manifold gate
%     Stage 2 : CNN_reader + MLP_reader — 10-class digit classification
%     Stage 3 : MD_Stage3_Postfilter (x2) — latent-space MD on CNN and MLP
%                 Algorithm 1  LHL    — last hidden layer
%                 Algorithm 2  FUSION — multi-layer feature concatenation
%                 Algorithm 3  MBM    — multi-branch, one detector per branch
%
%   Stage 3 network IDs and layer configs
%     CNN  id='CNN_8_16_32_fc64'
%          LHL    : relu4
%          FUSION : {relu1, relu2, relu3, relu4}
%          MBM    : {relu1} | {relu2} | {relu3, relu4}   (pooling boundaries)
%     MLP  id='MLP_512_256_128'
%          LHL    : relu3
%          FUSION : {relu1, relu2, relu3}
%          MBM    : {relu1} | {relu2} | {relu3}           (one branch per layer)

	% ── Resolve paths ────────────────────────────────────────────────────
	if nargin < 1 || strlength(string(trainFolder)) == 0
		trainRoot = getSetFolderPaths('resolve', 'trainRoot');
	else
		trainRoot = getSetFolderPaths('resolve', 'trainRoot', trainFolder);
	end
	if nargin < 2 || strlength(string(testFolder)) == 0
		oodFolder = getSetFolderPaths('resolve', 'testRoot');
	else
		oodFolder = getSetFolderPaths('resolve', 'testRoot', testFolder);
	end
	if nargin < 3 || isempty(rejectThreshold)
		rejectThreshold = 0.5;
	end
	if nargin < 4 || isempty(stage1Active)
		stage1Active = false;   % dormant by default — set true to enable filtering
	end
	if rejectThreshold < 0 || rejectThreshold > 1
		error('Folder_testor:badThreshold', 'rejectThreshold must be between 0 and 1.');
	end

	% When Stage 1 is dormant, force vigilance to 0 so all samples pass.
	activeVigilance = rejectThreshold * double(stage1Active);

	% ── Stage 3 layer configurations ─────────────────────────────────────
	% CNN: conv1(8)→relu1→pool1 | conv2(16)→relu2→pool2 | conv3(32)→relu3, fc1(64)→relu4
	cnnID  = 'CNN_8_16_32_fc64';
	cnnCfg = struct( ...
		'LHL',    'relu4', ...
		'FUSION', {{'relu1','relu2','relu3','relu4'}}, ...
		'MBM',    {{{'relu1'}, {'relu2'}, {'relu3','relu4'}}});

	% MLP: 784→512(relu1)→256(relu2)→128(relu3)→10
	mlpID  = 'MLP_512_256_128';
	mlpCfg = struct( ...
		'LHL',    'relu3', ...
		'FUSION', {{'relu1','relu2','relu3'}}, ...
		'MBM',    {{{'relu1'}, {'relu2'}, {'relu3'}}});

	algos  = {'LHL', 'FUSION', 'MBM'};
	aLabel = {'Algorithm 1 — LHL    (last hidden layer)', ...
	          'Algorithm 2 — FUSION (multi-layer concat)', ...
	          'Algorithm 3 — MBM    (multi-branch)'};

	% ── Header ───────────────────────────────────────────────────────────
	sep('=', 64);
	fprintf('  Folder Testor\n');
	fprintf('  Train : %s\n', getSetFolderPaths(trainRoot));
	fprintf('  Test  : %s\n', getSetFolderPaths(oodFolder));
	sep('=', 64);

	% ── Preallocate outputs ───────────────────────────────────────────────
	cnnYPred          = categorical.empty(0, 1);
	mlpYPred          = categorical.empty(0, 1);
	cnnResults        = struct('Network', [], 'Accuracy', NaN);
	mlpResults        = struct('Network', [], 'Accuracy', NaN);
	agreementAccepted = NaN;
	distTbl = table(string(0:9)', zeros(10,1), zeros(10,1), ...
		'VariableNames', {'PredictedDigit', 'CNN_Count', 'MLP_Count'});
	stage3.CNN = struct('LHL', [], 'FUSION', [], 'MBM', []);
	stage3.MLP = struct('LHL', [], 'FUSION', [], 'MBM', []);

	% =====================================================================
	if stage1Active
		stageHeader(1, 'Pixel-Space Pre-Filter', 'MD_Stage1_Prefilter');
	else
		stageHeader(1, 'Pixel-Space Pre-Filter  [DORMANT — all pass]', 'MD_Stage1_Prefilter');
	end
	% =====================================================================

	s1     = MD_Stage1_Prefilter(oodFolder, activeVigilance, false, false);
	nTotal = numel(s1.Accepted);

	sep('-', 64);
	if stage1Active
		fprintf('  Vigilance : %.2f\n', rejectThreshold);
	else
		fprintf('  Vigilance : 0.00  (dormant)\n');
	end
	fprintf('  Input     : %d samples\n', nTotal);
	fprintf('  Accepted  : %d  (%.1f%%)\n', s1.AcceptedCount, pct(s1.AcceptedCount, nTotal));
	fprintf('  Rejected  : %d  (%.1f%%)\n', s1.RejectedCount, pct(s1.RejectedCount, nTotal));
	sep('-', 64);

	if s1.AcceptedCount == 0
		stageHeader(2, 'Classification', 'CNN_reader + MLP_reader');
		fprintf('  No samples passed Stage 1 — skipping Stages 2 and 3.\n');
		sep('-', 64);
	else
		% =================================================================
		stageHeader(2, 'Classification', 'CNN_reader + MLP_reader');
		% =================================================================

		cnnResults = CNN_reader(trainRoot);
		mlpResults = MLP_reader(trainRoot);

		acceptedMask     = s1.Accepted;
		acceptedImages   = s1.Images4D(:, :, :, acceptedMask);   % for CNN
		acceptedFeatures = s1.Features(acceptedMask, :);          % for MLP

		cnnYPred = classify(cnnResults.Network, acceptedImages,   'MiniBatchSize', 256);
		mlpYPred = classify(mlpResults.Network, acceptedFeatures, 'MiniBatchSize', 512);

		agreementAccepted = mean(cnnYPred == mlpYPred) * 100;

		predCats  = categorical(string(0:9));
		cnnCounts = countcats(categorical(cellstr(string(cnnYPred)), cellstr(string(predCats))));
		mlpCounts = countcats(categorical(cellstr(string(mlpYPred)), cellstr(string(predCats))));
		distTbl   = table(string(0:9)', cnnCounts(:), mlpCounts(:), ...
			'VariableNames', {'PredictedDigit', 'CNN_Count', 'MLP_Count'});

		sep('-', 64);
		fprintf('  Samples classified : %d\n',    s1.AcceptedCount);
		fprintf('  CNN accuracy       : %.2f%%\n', cnnResults.Accuracy);
		fprintf('  MLP accuracy       : %.2f%%\n', mlpResults.Accuracy);
		fprintf('  CNN vs MLP agree   : %.2f%%\n', agreementAccepted);
		sep('-', 64);
		fprintf('  Digit distribution (accepted samples):\n');
		disp(distTbl);

		% =================================================================
		stageHeader(3, 'Latent-Space Post-Filter', 'MD_Stage3_Postfilter');
		% =================================================================

		nAcc = s1.AcceptedCount;

		% Run Stage 3 on each network
		nets = { ...
			struct('tag','CNN', 'net',cnnResults.Network, 'X',acceptedImages,   'id',cnnID, 'cfg',cnnCfg), ...
			struct('tag','MLP', 'net',mlpResults.Network, 'X',acceptedFeatures, 'id',mlpID, 'cfg',mlpCfg)};

		for n = 1:numel(nets)
			nn = nets{n};
			fprintf('\n');
			sep('-', 64);
			fprintf('  %s  (networkID: %s)\n', nn.tag, nn.id);
			sep('-', 64);

			for i = 1:numel(algos)
				algo = algos{i};
				lc   = nn.cfg.(algo);
				fprintf('\n  %s\n', aLabel{i});

				s3model = MD_Stage3_Postfilter('train', nn.net, nn.id, algo, lc);
				s3res   = MD_Stage3_Postfilter('test',  nn.net, nn.X,  s3model);

				stage3.(nn.tag).(algo) = s3res;
				printS3Result(s3res, algo, nAcc);
			end
		end
	end

	% ── Summary ──────────────────────────────────────────────────────────
	sep('=', 64);
	fprintf('  SUMMARY\n');
	sep('-', 64);
	fprintf('  %-38s : %d\n', 'Input samples', nTotal);
	fprintf('  %-38s : %d  (%.1f%%)\n', ...
		'After Stage 1', s1.AcceptedCount, pct(s1.AcceptedCount, nTotal));

	if s1.AcceptedCount > 0
		for n = 1:2
			tags = {'CNN','MLP'};
			tag  = tags{n};
			for i = 1:numel(algos)
				algo = algos{i};
				s3r  = stage3.(tag).(algo);
				if ~isempty(s3r)
					label = sprintf('After Stage 3  %s [%s]', tag, algo);
					fprintf('  %-38s : %d  (%.1f%% of input)\n', ...
						label, s3r.AcceptedCount, pct(s3r.AcceptedCount, nTotal));
				end
			end
		end
	end
	sep('=', 64);

	% ── Return struct ─────────────────────────────────────────────────────
	results = struct();
	results.TrainDataRoot        = trainRoot;
	results.OODDataRoot          = oodFolder;
	results.RejectThreshold      = rejectThreshold;
	results.Stage1               = s1;
	results.CNN                  = struct('YPred', cnnYPred, 'Results', cnnResults);
	results.MLP                  = struct('YPred', mlpYPred, 'Results', mlpResults);
	results.Stage3               = stage3;
	results.YOOD                 = s1.InputLabels;
	results.AgreementAcceptedPct = agreementAccepted;
	results.DigitDistribution    = distTbl;
	results.AcceptedMask         = s1.Accepted;
	results.MDFilter             = s1;   % legacy alias
end

% =========================================================================
% LOCAL HELPERS
% =========================================================================

function printS3Result(s3res, algo, nAcc)
	if strcmpi(algo, 'MBM')
		for b = 1:size(s3res.BranchIsOOD, 2)
			fprintf('    Branch %d  OOD flagged: %d\n', b, sum(s3res.BranchIsOOD(:,b)));
		end
		fprintf('    Accepted : %d / %d  (%.1f%% of Stage-1 accepted)\n', ...
			s3res.AcceptedCount, nAcc, pct(s3res.AcceptedCount, nAcc));
		fprintf('    Rejected : %d / %d  (%.1f%%)\n', ...
			s3res.RejectedCount, nAcc, pct(s3res.RejectedCount, nAcc));
	else
		fprintf('    Threshold : %.4f\n', s3res.Threshold);
		fprintf('    Accepted  : %d / %d  (%.1f%% of Stage-1 accepted)\n', ...
			s3res.AcceptedCount, nAcc, pct(s3res.AcceptedCount, nAcc));
		fprintf('    Rejected  : %d / %d  (%.1f%%)\n', ...
			s3res.RejectedCount, nAcc, pct(s3res.RejectedCount, nAcc));
	end
end

function stageHeader(n, title, func)
	sep('=', 64);
	fprintf('  STAGE %d  |  %s\n', n, title);
	fprintf('           |  %s\n', func);
	sep('=', 64);
end

function sep(ch, w)
	fprintf('%s\n', repmat(ch, 1, w));
end

function v = pct(num, den)
	if den == 0, v = 0; else, v = 100 * num / den; end
end
