function insert_donut_artifact(srcFolder, dstFolder, numImages, outerDia, innerDia, greyLevel)
% insert_donut_artifact  Copy randomly-drawn chest X-rays and stamp a grey donut.
%
%   insert_donut_artifact() uses defaults:
%     srcFolder  = <project>/chex_train
%     dstFolder  = <project>/chex_donut75
%     numImages  = 3000
%     outerDia   = 75   (outer diameter in pixels)
%     innerDia   = 25   (hole diameter in pixels)
%     greyLevel  = 128  (0-255 mid-grey)
%
%   The donut mask is a filled disc of radius outerDia/2 with a circular
%   hole of radius innerDia/2 cut from its centre.  Centre positions are
%   drawn from a 2-D Gaussian centred on the image centre (sigma = H/3)
%   so most placements are near the middle while a meaningful tail falls
%   close to or partially past an edge.  The centre is clamped so the
%   bounding box of the outer disc stays within the image.
%
%   Usage
%   -----
%   insert_donut_artifact()
%   insert_donut_artifact(srcFolder, dstFolder, 1000, 75, 25, 128)

    % ------------------------------------------------------------------ %
    %  Defaults
    % ------------------------------------------------------------------ %
    here = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(here);

    if nargin < 1 || isempty(srcFolder)
        srcFolder = fullfile(projectRoot, 'chex_train');
    end
    if nargin < 2 || isempty(dstFolder)
        dstFolder = fullfile(projectRoot, 'chex_donut75');
    end
    if nargin < 3 || isempty(numImages)
        numImages = 3000;
    end
    if nargin < 4 || isempty(outerDia)
        outerDia = 75;
    end
    if nargin < 5 || isempty(innerDia)
        innerDia = 25;
    end
    if nargin < 6 || isempty(greyLevel)
        greyLevel = 128;
    end

    outerR = outerDia / 2;
    innerR = innerDia / 2;

    % ------------------------------------------------------------------ %
    %  Pre-compute the donut mask (bounding box = outerDia x outerDia)
    % ------------------------------------------------------------------ %
    % Pixel grid relative to the centre of the bounding box
    bbox   = outerDia;                         % bounding box side length
    half   = floor(bbox / 2);
    [gc, gr] = meshgrid(-half:half, -half:half);  % col offset, row offset
    distSq = gr.^2 + gc.^2;
    donutMask = (distSq <= outerR^2) & (distSq > innerR^2);

    % ------------------------------------------------------------------ %
    %  Gather source files
    % ------------------------------------------------------------------ %
    exts     = {'.jpg', '.jpeg', '.png'};
    allFiles = [];
    for e = 1:numel(exts)
        hits     = dir(fullfile(srcFolder, ['*' exts{e}]));
        allFiles = [allFiles; hits]; %#ok<AGROW>
    end

    if isempty(allFiles)
        error('insert_donut_artifact:noImages', ...
              'No .jpg/.jpeg/.png images found in: %s', srcFolder);
    end

    N = numel(allFiles);
    if numImages > N
        warning('insert_donut_artifact:tooFew', ...
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
    greyU8   = uint8(greyLevel);
    maskRows = size(donutMask, 1);   % = outerDia + 1  (meshgrid symmetric)
    maskCols = size(donutMask, 2);

    fprintf('Inserting donut (outer=%dpx, inner=%dpx, level=%d) into %d images...\n', ...
            outerDia, innerDia, greyLevel, numImages);

    for k = 1:numImages
        srcPath = fullfile(allFiles(idx(k)).folder, allFiles(idx(k)).name);
        img     = imread(srcPath);

        % Ensure grayscale
        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        [H, W] = size(img);

        % Gaussian centre placement, sigma = H/3 (same as square utility)
        sigmaR = H / 3;
        sigmaC = W / 3;

        cx = round(H/2 + sigmaR * randn());   % row centre
        cy = round(W/2 + sigmaC * randn());   % col centre

        % Clamp so the bounding box stays inside the image
        cx = max(half + 1, min(cx, H - half));
        cy = max(half + 1, min(cy, W - half));

        % Bounding-box row/col ranges in the image
        rRows = (cx - half):(cx - half + maskRows - 1);
        rCols = (cy - half):(cy - half + maskCols - 1);

        % Stamp only the donut pixels (leave hole and exterior untouched)
        patch = img(rRows, rCols);
        patch(donutMask) = greyU8;
        img(rRows, rCols) = patch;

        % Save to destination (keep original filename)
        dstPath = fullfile(dstFolder, allFiles(idx(k)).name);
        imwrite(img, dstPath);

        if mod(k, 500) == 0
            fprintf('  %d / %d done\n', k, numImages);
        end
    end

    fprintf('Done. %d modified images saved to:\n  %s\n', numImages, dstFolder);
end
