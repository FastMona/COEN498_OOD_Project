function fig = visualize_chex_examples(testFolder, vigilance, maxPerGroup, stage)
% visualize_chex_examples  Show accepted and rejected CheXpert samples side by side.
%
%   visualize_chex_examples() runs MD_chex on chex_pacemaker with vigilance
%   0.5 and displays a compact grid of accepted and rejected examples.
%
%   visualize_chex_examples(testFolder, vigilance, maxPerGroup, stage)
%     testFolder  : folder of .jpg chest X-rays to test (default: chex_pacemaker)
%     vigilance   : MD acceptance threshold in [0,1] (default: 0.5)
%     maxPerGroup : max images shown per row (default: 5, capped at 5)
%     stage       : 1 = Stage-1 pixel MD, 3 = Stage-3 latent MD (default: 1)

	here        = fileparts(mfilename('fullpath'));
	projectRoot = fileparts(here);

	if nargin < 1 || strlength(string(testFolder)) == 0
		% Silently read the last-used testRoot from the getSetFolderPaths cache
		% so the visualization stays in sync with the most recent Chex_tester run.
		testFolder = readStoredTestRoot(projectRoot);
	else
		% Resolve any relative path to absolute so MD_chex can always find it.
		testFolder = resolveToAbsolute(char(string(testFolder)), projectRoot);
	end
	if nargin < 2 || isempty(vigilance)
		vigilance = 0.5;
	end
	if nargin < 3 || isempty(maxPerGroup)
		maxPerGroup = 5;
	end
	if nargin < 4 || isempty(stage)
		stage = 1;
	end
	maxPerGroup = min(maxPerGroup, 5);

	results = MD_chex(testFolder, [], vigilance, false);

	if stage == 3
		acceptedIdx = find(results.LatentAccepted);
		rejectedIdx = find(results.Accepted & ~results.LatentAccepted);
		stageLabel  = 'Stage 3: latent MD';
	else
		acceptedIdx = find(results.Accepted);
		rejectedIdx = find(results.IsOOD);
		stageLabel  = 'Stage 1: pixel MD';
	end

	numAccepted = min(maxPerGroup, numel(acceptedIdx));
	numRejected = min(maxPerGroup, numel(rejectedIdx));
	numCols = 5;

	fig = figure('Name', 'CheXpert Filter Visualization', 'Color', 'w');
	t = tiledlayout(2, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
	title(t, sprintf('MD\_chex  %s  at vigilance %.2f', stageLabel, vigilance));

	for i = 1:numCols
		nexttile(i);
		if i <= numAccepted
			showExample(results, acceptedIdx(i), true, stage);
		else
			axis off;
		end
	end

	for i = 1:numCols
		nexttile(numCols + i);
		if i <= numRejected
			showExample(results, rejectedIdx(i), false, stage);
		else
			axis off;
		end
	end

	annotation(fig, 'textbox', [0 0.95 1 0.04], ...
		'String', sprintf('Accepted: %d    Rejected: %d    Source: %s', ...
			numel(acceptedIdx), numel(rejectedIdx), testFolder), ...
		'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

function folder = readStoredTestRoot(projectRoot)
% Silently return the cached testRoot, falling back to chex_pacemaker.
	cacheFile = fullfile(projectRoot, 'trained_models', 'folder_paths_cache.mat');
	if isfile(cacheFile)
		data = load(cacheFile, 'folderPaths');
		if isfield(data, 'folderPaths') && isfield(data.folderPaths, 'testRoot')
			stored = char(string(data.folderPaths.testRoot));
			if ~isempty(stored) && isfolder(stored)
				folder = stored;
				return;
			end
		end
	end
	folder = fullfile(projectRoot, 'chex_pacemaker');
end

function absPath = resolveToAbsolute(pathInput, projectRoot)
% Convert a relative path to absolute using projectRoot as the base.
	isAbs = (numel(pathInput) >= 2 && pathInput(2) == ':') || ...
	        startsWith(pathInput, '\\');
	if isAbs
		absPath = pathInput;
	else
		absPath = fullfile(projectRoot, pathInput);
	end
end

function showExample(results, sampleIdx, isAccepted, stage)
	img = imread(results.TestFiles{sampleIdx});
	if size(img, 3) == 3
		img = rgb2gray(img);
	end
	imshow(img, []);
	axis image off;

	if isAccepted
		status = 'ACCEPT';
	else
		status = 'REJECT';
	end

	if stage == 3
		titleLines = { ...
			sprintf('%s  combined=%.3f', status, results.CombinedScores(sampleIdx)), ...
			sprintf('latent=%.3f  dist=%.3f', results.LatentScores(sampleIdx), ...
				results.NormalizedDist(sampleIdx)) ...
		};
	else
		titleLines = { ...
			sprintf('%s  md=%.3f', status, results.MDScores(sampleIdx)), ...
			sprintf('dist=%.3f', results.NormalizedDist(sampleIdx)) ...
		};
	end

	title(titleLines, 'FontSize', 9);
end
