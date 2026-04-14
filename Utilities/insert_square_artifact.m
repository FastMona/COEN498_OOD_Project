function insert_square_artifact(srcFolder, dstFolder, numImages, squareSize, greyLevel)
% insert_square_artifact  Copy randomly-drawn chest X-rays and stamp a grey square.
%
%   insert_square_artifact() uses defaults:
%     srcFolder  = <project>/chex_train
%     dstFolder  = <project>/chex_squares75
%     numImages  = 3000
%     squareSize = 75          (pixels, square is squareSize x squareSize)
%     greyLevel  = 128         (0-255 mid-grey)
%
%   Square centre positions are drawn from a 2-D Gaussian centred on the
%   image centre.  The standard deviation is set to (imageHeight / 3) so
%   the bulk of placements fall well inside the frame while a meaningful
%   tail lands near or partially outside an edge (the square is clamped to
%   [1, imageDim] so it is always fully within the image boundary).
%
%   Usage
%   -----
%   insert_square_artifact()
%   insert_square_artifact(srcFolder, dstFolder, 1000, 75, 100)

    % ------------------------------------------------------------------ %
    %  Defaults
    % ------------------------------------------------------------------ %
    here = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(here);

    if nargin < 1 || isempty(srcFolder)
        srcFolder = fullfile(projectRoot, 'chex_train');
    end
    if nargin < 2 || isempty(dstFolder)
        dstFolder = fullfile(projectRoot, 'chex_squares75');
    end
    if nargin < 3 || isempty(numImages)
        numImages = 3000;
    end
    if nargin < 4 || isempty(squareSize)
        squareSize = 75;
    end
    if nargin < 5 || isempty(greyLevel)
        greyLevel  = 128;
    end

    % ------------------------------------------------------------------ %
    %  Gather source files
    % ------------------------------------------------------------------ %
    exts   = {'.jpg', '.jpeg', '.png'};
    allFiles = [];
    for e = 1:numel(exts)
        hits = dir(fullfile(srcFolder, ['*' exts{e}]));
        allFiles = [allFiles; hits]; %#ok<AGROW>
    end

    if isempty(allFiles)
        error('insert_square_artifact:noImages', ...
              'No .jpg/.jpeg/.png images found in: %s', srcFolder);
    end

    N = numel(allFiles);
    if numImages > N
        warning('insert_square_artifact:tooFew', ...
                'Requested %d images but only %d available – using all.', numImages, N);
        numImages = N;
    end

    % Random draw without replacement
    idx = randperm(N, numImages);

    % ------------------------------------------------------------------ %
    %  Create destination folder
    % ------------------------------------------------------------------ %
    if ~exist(dstFolder, 'dir')
        mkdir(dstFolder);
        fprintf('Created destination folder: %s\n', dstFolder);
    end

    % ------------------------------------------------------------------ %
    %  Process images
    % ------------------------------------------------------------------ %
    half   = floor(squareSize / 2);
    greyU8 = uint8(greyLevel);

    fprintf('Inserting %dx%d grey square (level=%d) into %d images...\n', ...
            squareSize, squareSize, greyLevel, numImages);

    for k = 1:numImages
        srcPath = fullfile(allFiles(idx(k)).folder, allFiles(idx(k)).name);
        img     = imread(srcPath);

        % Ensure grayscale (some CheXpert JPEGs are stored as RGB)
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        [H, W] = size(img);

        % Gaussian centre placement
        % sigma = H/3 gives ~5 % of draws outside ±H/2 (near / past edge)
        sigmaR = H / 3;
        sigmaC = W / 3;

        cx = round(H/2 + sigmaR * randn());   % row centre
        cy = round(W/2 + sigmaC * randn());   % col centre

        % Clamp so the full square stays inside the image
        cx = max(half + 1, min(cx, H - half));
        cy = max(half + 1, min(cy, W - half));

        % Stamp the square
        rRows = (cx - half):(cx - half + squareSize - 1);
        rCols = (cy - half):(cy - half + squareSize - 1);
        img(rRows, rCols) = greyU8;

        % Save to destination (keep original filename)
        dstPath = fullfile(dstFolder, allFiles(idx(k)).name);
        imwrite(img, dstPath);

        if mod(k, 500) == 0
            fprintf('  %d / %d done\n', k, numImages);
        end
    end

    fprintf('Done. %d modified images saved to:\n  %s\n', numImages, dstFolder);
end
