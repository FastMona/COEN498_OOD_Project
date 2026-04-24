function [X, Y] = loadMNISTForMLP(dataFolder)
% loadMNISTForMLP  Load MNIST IDX files into the format expected by testMLP.
%
%   [X, Y] = loadMNISTForMLP(dataFolder)
%
%   Returns:
%     X — 784 × N single matrix, pixel values in [0,1]
%     Y — N × 1 double, labels in 1–10  (0→1, …, 9→10)
%
%   Tries the test split first (t10k-*), then the training split.

    candidates = { ...
        {'t10k-images-idx3-ubyte',   't10k-labels-idx1-ubyte'}, ...
        {'train-images-idx3-ubyte',  'train-labels-idx1-ubyte'}};

    imgFile = '';
    lblFile = '';
    for i = 1:numel(candidates)
        f = fullfile(dataFolder, candidates{i}{1});
        if isfile(f)
            imgFile = f;
            lblFile = fullfile(dataFolder, candidates{i}{2});
            break;
        end
    end
    if isempty(imgFile)
        error('loadMNISTForMLP:notFound', 'No IDX image file found in: %s', dataFolder);
    end

    fprintf('Loading images : %s\n', imgFile);
    images4D = readIDXImages(imgFile);
    labels   = readIDXLabels(lblFile);

    [H, W, ~, N] = size(images4D);
    X = reshape(images4D, H*W, N);   % 784 × N, already [0,1]
    Y = double(labels) + 1;          % 0–9 → 1–10
end

function images4D = readIDXImages(filePath)
    fid = fopen(filePath, 'rb');
    if fid < 0, error('Cannot open: %s', filePath); end
    cleaner = onCleanup(@() fclose(fid));
    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magic ~= 2051, error('Invalid IDX magic in %s (got %d)', filePath, magic); end
    numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
    numRows   = fread(fid, 1, 'int32', 0, 'ieee-be');
    numCols   = fread(fid, 1, 'int32', 0, 'ieee-be');
    pixels    = fread(fid, numImages*numRows*numCols, 'uint8=>single');
    img3D     = reshape(pixels, [numCols, numRows, numImages]);
    img3D     = permute(img3D, [2,1,3]) ./ 255;
    images4D  = reshape(img3D, [numRows, numCols, 1, numImages]);
end

function labels = readIDXLabels(filePath)
    fid = fopen(filePath, 'rb');
    if fid < 0, error('Cannot open: %s', filePath); end
    cleaner = onCleanup(@() fclose(fid));
    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magic ~= 2049, error('Invalid IDX magic in %s (got %d)', filePath, magic); end
    numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
    labels    = fread(fid, numLabels, 'uint8=>double');
end
