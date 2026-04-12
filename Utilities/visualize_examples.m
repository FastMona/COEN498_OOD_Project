function fig = visualize_examples(inputFolder)
% visualize_examples Display 10 random patterns from an IDX dataset folder.
%
%   visualize_examples() loads samples from MNIST_digits (or MNIST_digits/raw
%   if present) and displays 10 random patterns in a 2x5 grid.
%
%   visualize_examples(inputFolder) uses the provided folder.

	if nargin < 1 || strlength(string(inputFolder)) == 0
		here = fileparts(mfilename('fullpath'));
		inputFolder = fullfile(here, 'MNIST_fashion');
	end

	datasetFolder = resolveDatasetFolder(char(string(inputFolder)));
	imagesPath = fullfile(datasetFolder, 't10k-images-idx3-ubyte');
	labelsPath = fullfile(datasetFolder, 't10k-labels-idx1-ubyte');

	if ~isfile(imagesPath)
		error('visualize_examples:missingImages', 'Missing file: %s', imagesPath);
	end

	images4D = readIDXImages(imagesPath);
	numSamples = size(images4D, 4);
	if numSamples == 0
		error('visualize_examples:noSamples', 'No samples found in: %s', imagesPath);
	end

	hasLabels = isfile(labelsPath);
	if hasLabels
		labels = readIDXLabels(labelsPath);
	else
		labels = [];
	end

	numToShow = min(10, numSamples);
	perm = randperm(numSamples, numToShow);

	fig = figure('Name', 'Random Pattern Visualization', 'Color', 'w');
	t = tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
	title(t, sprintf('Random samples from: %s', datasetFolder), 'Interpreter', 'none');

	for i = 1:10
		nexttile(i);
		if i <= numToShow
			idx = perm(i);
			imshow(images4D(:, :, 1, idx), []);
			axis image off;
			if hasLabels && numel(labels) >= idx
				title(sprintf('idx=%d  label=%d', idx, labels(idx)), 'FontSize', 9);
			else
				title(sprintf('idx=%d', idx), 'FontSize', 9);
			end
		else
			axis off;
		end
	end
end

function datasetFolder = resolveDatasetFolder(inputFolder)
	rawCandidate = fullfile(inputFolder, 'raw');
	if isfolder(rawCandidate)
		datasetFolder = rawCandidate;
		return;
	end

	if isfolder(inputFolder)
		datasetFolder = inputFolder;
		return;
	end

	error('visualize_examples:missingFolder', 'Folder not found: %s', inputFolder);
end

function images4D = readIDXImages(filePath)
	fid = fopen(filePath, 'rb');
	if fid < 0
		error('visualize_examples:openFailed', 'Could not open image file: %s', filePath);
	end
	cleanupObj = onCleanup(@() fclose(fid));

	magic = fread(fid, 1, 'uint32', 0, 'ieee-be');
	if magic ~= 2051
		error('visualize_examples:badMagic', 'Invalid image IDX magic in %s', filePath);
	end

	numImages = fread(fid, 1, 'uint32', 0, 'ieee-be');
	numRows = fread(fid, 1, 'uint32', 0, 'ieee-be');
	numCols = fread(fid, 1, 'uint32', 0, 'ieee-be');

	raw = fread(fid, numRows * numCols * numImages, 'uint8=>single');
	if numel(raw) ~= numRows * numCols * numImages
		error('visualize_examples:truncatedImages', 'Image IDX appears truncated: %s', filePath);
	end

	images = reshape(raw, [numCols, numRows, numImages]);
	images = permute(images, [2, 1, 3]);
	images4D = reshape(images, numRows, numCols, 1, numImages) / 255;
end

function labels = readIDXLabels(filePath)
	fid = fopen(filePath, 'rb');
	if fid < 0
		error('visualize_examples:openFailed', 'Could not open label file: %s', filePath);
	end
	cleanupObj = onCleanup(@() fclose(fid));

	magic = fread(fid, 1, 'uint32', 0, 'ieee-be');
	if magic ~= 2049
		error('visualize_examples:badMagic', 'Invalid label IDX magic in %s', filePath);
	end

	numLabels = fread(fid, 1, 'uint32', 0, 'ieee-be');
	labels = fread(fid, numLabels, 'uint8=>double');
	if numel(labels) ~= numLabels
		error('visualize_examples:truncatedLabels', 'Label IDX appears truncated: %s', filePath);
	end
end
