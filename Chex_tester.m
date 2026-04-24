function results = Chex_tester(chexRoot, testFolder, rejectThreshold, stage1Active)
% Chex_tester  Three-stage OOD pipeline for CheXpert chest X-rays.
%
%   Mirrors Folder_testor but for CheXpert (one-class, JPEG images):
%
%   Pipeline
%     Stage 1 : MD1_chex              — pixel-space PCA manifold gate
%     Stage 2 : CNN_chex + MLP_chex   — regression  (normal target = 1.0)
%     Stage 3 : MD3_chex (×2 nets, ×3 algos) — latent-space MD filter
%                 Algorithm 1  LHL    — last hidden layer
%                 Algorithm 2  FUSION — multi-layer feature concatenation
%                 Algorithm 3  MBM    — multi-branch, one detector per branch
%
%   Stage 3 network IDs and layer configs
%     CNN  id='CNN_chex_16_32_64_128_fc256'
%          LHL    : relu5
%          FUSION : {relu1, relu2, relu3, relu4, relu5}
%          MBM    : {relu1} | {relu2} | {relu3} | {relu4, relu5}
%     MLP  id='MLP_chex_1024_512_256_128'
%          LHL    : relu4
%          FUSION : {relu1, relu2, relu3, relu4}
%          MBM    : {relu1} | {relu2} | {relu3} | {relu4}
%
%   RESULTS = Chex_tester()
%   RESULTS = Chex_tester(chexRoot, testFolder)
%   RESULTS = Chex_tester(chexRoot, testFolder, rejectThreshold)
%   RESULTS = Chex_tester(chexRoot, testFolder, rejectThreshold, stage1Active)
%
%     chexRoot        : flat folder of normal training .jpg images
%     testFolder      : flat folder of test .jpg images (ID or OOD)
%     rejectThreshold : Stage-1 vigilance [0,1]  default 0.5
%     stage1Active    : logical, default false (Stage 1 dormant — all pass)
%                       Set true to enable pixel-space filtering.

	% ── Resolve paths ────────────────────────────────────────────────────
	if nargin < 1 || isempty(chexRoot)
		chexRoot = getSetFolderPaths('resolve', 'trainRoot');
	else
		chexRoot = getSetFolderPaths('resolve', 'trainRoot', chexRoot);
	end
	if nargin < 2 || isempty(testFolder)
		testFolder = getSetFolderPaths('resolve', 'testRoot');
	else
		testFolder = getSetFolderPaths('resolve', 'testRoot', testFolder);
	end
	if nargin < 3 || isempty(rejectThreshold)
		rejectThreshold = 0.5;
	end
	if nargin < 4 || isempty(stage1Active)
		stage1Active = false;
	end
	if rejectThreshold < 0 || rejectThreshold > 1
		error('Chex_tester:badThreshold', 'rejectThreshold must be between 0 and 1.');
	end

	activeVigilance = rejectThreshold * double(stage1Active);

	% ── Stage 3 layer configurations ─────────────────────────────────────
	% CNN: conv1(16)→relu1→pool1 | conv2(32)→relu2→pool2 |
	%       conv3(64)→relu3→pool3 | conv4(128)→relu4→pool4 | fc1(256)→relu5
	cnnID  = 'CNN_chex_16_32_64_128_fc256';
	cnnCfg = struct( ...
		'LHL',    'relu5', ...
		'FUSION', {{'relu1','relu2','relu3','relu4','relu5'}}, ...
		'MBM',    {{{'relu1'},{'relu2'},{'relu3'},{'relu4','relu5'}}});

	% MLP: flatten→fc1(1024)→relu1→fc2(512)→relu2→fc3(256)→relu3→fc4(128)→relu4
	mlpID  = 'MLP_chex_1024_512_256_128';
	mlpCfg = struct( ...
		'LHL',    'relu4', ...
		'FUSION', {{'relu1','relu2','relu3','relu4'}}, ...
		'MBM',    {{{'relu1'},{'relu2'},{'relu3'},{'relu4'}}});

	algos  = {'LHL', 'FUSION', 'MBM'};
	aLabel = {'Algorithm 1 — LHL    (last hidden layer)', ...
	          'Algorithm 2 — FUSION (multi-layer concat)', ...
	          'Algorithm 3 — MBM    (multi-branch)'};

	% ── Header ───────────────────────────────────────────────────────────
	sep('=', 64);
	fprintf('  Chex Tester\n');
	fprintf('  Train : %s\n', getSetFolderPaths(chexRoot));
	fprintf('  Test  : %s\n', getSetFolderPaths(testFolder));
	sep('=', 64);

	% ── Preallocate outputs ───────────────────────────────────────────────
	cnnNet      = [];
	mlpNet      = [];
	cnnS2Scores = [];
	mlpS2Scores = [];
	stage3.CNN  = struct('LHL', [], 'FUSION', [], 'MBM', []);
	stage3.MLP  = struct('LHL', [], 'FUSION', [], 'MBM', []);

	% =====================================================================
	if stage1Active
		stageHeader(1, 'Pixel-Space Pre-Filter', 'MD1_chex');
	else
		stageHeader(1, 'Pixel-Space Pre-Filter  [DORMANT — all pass]', 'MD1_chex');
	end
	% =====================================================================

	s1     = MD1_chex(testFolder, chexRoot, activeVigilance, false);
	nTotal = s1.NumSamples;

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
		stageHeader(2, 'Regression', 'CNN_chex + MLP_chex');
		fprintf('  No samples passed Stage 1 — skipping Stages 2 and 3.\n');
		sep('-', 64);
	else
		% =================================================================
		stageHeader(2, 'Regression', 'CNN_chex + MLP_chex');
		% =================================================================

		cnnResults = CNN_chex(chexRoot);
		mlpResults = MLP_chex(chexRoot);
		cnnNet = cnnResults.Network;
		mlpNet = mlpResults.Network;

		% Subset: Stage-1 accepted images only
		acceptedFiles = s1.TestFiles(s1.Accepted);
		acceptedImds  = imageDatastore(acceptedFiles, 'ReadFcn', @readAndPreprocess);

		cnnS2Scores = predict(cnnNet, acceptedImds, 'MiniBatchSize', 32);
		reset(acceptedImds);
		mlpS2Scores = predict(mlpNet, acceptedImds, 'MiniBatchSize', 64);

		sep('-', 64);
		fprintf('  Samples scored     : %d\n',          s1.AcceptedCount);
		fprintf('  CNN score          : mean=%.4f  std=%.4f  (normal target=1.0)\n', ...
			mean(cnnS2Scores), std(cnnS2Scores));
		fprintf('  MLP score          : mean=%.4f  std=%.4f\n', ...
			mean(mlpS2Scores), std(mlpS2Scores));
		sep('-', 64);

		% =================================================================
		stageHeader(3, 'Latent-Space Post-Filter', 'MD3_chex');
		% =================================================================

		nAcc = s1.AcceptedCount;

		nets = { ...
			struct('tag','CNN', 'net',cnnNet, 'id',cnnID, 'cfg',cnnCfg), ...
			struct('tag','MLP', 'net',mlpNet, 'id',mlpID, 'cfg',mlpCfg)};

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

				s3model = MD3_chex('train', nn.net, nn.id, chexRoot, algo, lc);
				reset(acceptedImds);
				s3res   = MD3_chex('test',  nn.net, acceptedImds, s3model);

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
			tags = {'CNN', 'MLP'};
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
	results.ChexRoot        = chexRoot;
	results.TestFolder      = testFolder;
	results.RejectThreshold = rejectThreshold;
	results.Stage1          = s1;
	results.CNN             = struct('Stage2Scores', cnnS2Scores, 'Network', cnnNet);
	results.MLP             = struct('Stage2Scores', mlpS2Scores, 'Network', mlpNet);
	results.Stage3          = stage3;
	results.TestFiles       = s1.TestFiles;
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

function img = readAndPreprocess(filename)
	img = imread(filename);
	if size(img, 3) == 3
		img = rgb2gray(img);
	end
	img = im2single(img);
	if ismatrix(img)
		img = reshape(img, size(img, 1), size(img, 2), 1);
	end
end
