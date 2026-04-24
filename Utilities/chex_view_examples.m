function fig = chex_view_examples(defaultFolder)
% chex_view_examples  Browse a folder of CheXpert images interactively.
%
%   Prints the default folder, prompts the user to confirm or replace it,
%   then displays 10 randomly selected images in a 2x5 grid.
%
%   chex_view_examples()
%   chex_view_examples(defaultFolder)

    here        = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(here);

    % -----------------------------------------------------------------------
    % Resolve default folder — read cache silently to avoid triggering
    % the interactive prompt inside getSetFolderPaths.
    % -----------------------------------------------------------------------
    if nargin < 1 || isempty(defaultFolder)
        defaultFolder = readCachedPath(projectRoot, 'testRoot');
        if isempty(defaultFolder) || ~isfolder(defaultFolder)
            defaultFolder = fullfile(projectRoot, 'chex_train');
        end
    end

    % -----------------------------------------------------------------------
    % Prompt
    % -----------------------------------------------------------------------
    fprintf('\nLoading data from: %s\n', defaultFolder);
    reply = input('  Enter new folder path, or press Enter to accept: ', 's');

    if ~isempty(strtrim(reply))
        folder = strtrim(reply);
        isAbs  = (numel(folder) >= 2 && folder(2) == ':') || startsWith(folder, '\\');
        if ~isAbs
            folder = fullfile(projectRoot, folder);
        end
    else
        folder = defaultFolder;
    end

    if ~isfolder(folder)
        error('chex_view_examples:badFolder', 'Folder not found: %s', folder);
    end

    % -----------------------------------------------------------------------
    % Find images
    % -----------------------------------------------------------------------
    jpgFiles  = dir(fullfile(folder, '*.jpg'));
    jpegFiles = dir(fullfile(folder, '*.jpeg'));
    files = [jpgFiles; jpegFiles];

    if isempty(files)
        error('chex_view_examples:noImages', 'No .jpg files found in: %s', folder);
    end

    N     = numel(files);
    nShow = min(10, N);
    fprintf('  Found %d images — showing %d at random.\n\n', N, nShow);

    idx = randperm(N, nShow);

    % -----------------------------------------------------------------------
    % Display 2x5 grid
    % -----------------------------------------------------------------------
    [~, folderName] = fileparts(folder);
    fig = figure('Name', sprintf('chex_view_examples: %s', folderName), 'Color', 'w');
    tl  = tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, folder, 'Interpreter', 'none', 'FontSize', 8, 'FontWeight', 'bold');

    for i = 1:nShow
        nexttile;
        filepath = fullfile(files(idx(i)).folder, files(idx(i)).name);
        img = imread(filepath);
        if size(img, 3) == 3, img = rgb2gray(img); end
        imshow(img, []);
        title(files(idx(i)).name, 'FontSize', 7, 'Interpreter', 'none');
    end
    for i = nShow + 1:10
        nexttile;  axis off;
    end
end

function folder = readCachedPath(projectRoot, key)
% Read a stored folder path from the cache file without printing or prompting.
    folder = '';
    cacheFile = fullfile(projectRoot, 'trained_models', 'folder_paths_cache.mat');
    if ~isfile(cacheFile), return; end
    data = load(cacheFile, 'folderPaths');
    if isfield(data, 'folderPaths') && isfield(data.folderPaths, key)
        folder = char(string(data.folderPaths.(key)));
    end
end
