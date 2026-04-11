function fig = visualize_ood_examples(oodFolder, vigilance, maxPerGroup)
% visualize_ood_examples Show accepted and rejected OOD samples side by side.
%
%   visualize_ood_examples() runs MD_filter on KMNIST_japanese with vigilance
%   0.5 and displays a compact grid of accepted and rejected examples.
%
%   visualize_ood_examples(oodFolder, vigilance, maxPerGroup) lets you
%   choose the source folder, threshold, and maximum number of examples shown
%   from each group.

	if nargin < 1 || strlength(string(oodFolder)) == 0
		here = fileparts(mfilename('fullpath'));
		oodFolder = fullfile(here, 'KMNIST_japanese');
	end
	if nargin < 2 || isempty(vigilance)
		vigilance = 0.5;
	end
	if nargin < 3 || isempty(maxPerGroup)
		maxPerGroup = 5;
	end
	maxPerGroup = min(maxPerGroup, 5);

	results = MD_filter(oodFolder, vigilance, false, false);
	acceptedIdx = find(results.Accepted);
	rejectedIdx = find(results.IsOOD);

	numAccepted = min(maxPerGroup, numel(acceptedIdx));
	numRejected = min(maxPerGroup, numel(rejectedIdx));
	numCols = 5;

	fig = figure('Name', 'OOD Filter Visualization', 'Color', 'w');
	t = tiledlayout(2, numCols, 'TileSpacing', 'compact', 'Padding', 'compact');
	title(t, sprintf('MD\_filter at vigilance %.2f', vigilance));

	for i = 1:numCols
		nexttile(i);
		if i <= numAccepted
			showExample(results, acceptedIdx(i), true);
		else
			axis off;
		end
	end

	for i = 1:numCols
		nexttile(numCols + i);
		if i <= numRejected
			showExample(results, rejectedIdx(i), false);
		else
			axis off;
		end
	end

	annotation(fig, 'textbox', [0 0.95 1 0.04], ...
		'String', sprintf('Accepted: %d    Rejected: %d    Source: %s', ...
			results.AcceptedCount, results.RejectedCount, oodFolder), ...
		'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

function showExample(results, sampleIdx, isAccepted)
	img = results.Images4D(:, :, 1, sampleIdx);
	imshow(img, []);
	axis image off;

	status = 'REJECT';
	if isAccepted
		status = 'ACCEPT';
	end

	title({ ...
		sprintf('%s  pred=%d', status, results.BestDigit(sampleIdx)), ...
		sprintf('conf=%.3f  dist=%.3f', ...
			results.Confidence(sampleIdx), results.BestDistance(sampleIdx)) ...
	}, 'FontSize', 9);
end