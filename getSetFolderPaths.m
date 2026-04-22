function out = getSetFolderPaths(varargin)
% getSetFolderPaths Resolve, store, and display project folder paths.
%
%   cfg = getSetFolderPaths()
%     Resolves both trainRoot and testRoot,
%     stores them, and returns a struct with both absolute paths.
%
%   relPath = getSetFolderPaths(fullPath)
%     Converts fullPath to a project-relative display path when possible.
%
%   folderPath = getSetFolderPaths('resolve', key)
%   folderPath = getSetFolderPaths('resolve', key, providedPath)
%     Resolves a folder path by priority:
%       1) providedPath (if non-empty)
%       2) previously stored value for key
%       3) key-specific default when no value is stored
%     The resolved value is saved under key.

	projectRoot = fileparts(mfilename('fullpath'));

	if nargin == 0
		trainRoot = getSetFolderPaths('resolve', 'trainRoot');
		testRoot = getSetFolderPaths('resolve', 'testRoot');
		out = struct('trainRoot', trainRoot, 'testRoot', testRoot);
		fprintf('loading training data from: %s\n', getSetFolderPaths(trainRoot));
		fprintf('Running inference test on: %s\n', getSetFolderPaths(testRoot));
		return;
	end

	if ischar(varargin{1}) || (isstring(varargin{1}) && isscalar(varargin{1}))
		firstArg = char(string(varargin{1}));
		if strcmpi(firstArg, 'resolve')
			out = resolveAndStorePath(projectRoot, varargin{:});
			return;
		end
	end

	% Backward-compatible display mode.
	out = toRelativePath(char(string(varargin{1})), projectRoot);
end

function resolvedPath = resolveAndStorePath(projectRoot, varargin)
	if nargin < 3
		error('getSetFolderPaths:badResolveInput', ...
			'Usage: getSetFolderPaths(''resolve'', key, [providedPath]).');
	end

	key = char(string(varargin{2}));
	defaultPath = getDefaultForKey(projectRoot, key);

	providedPath = '';
	if numel(varargin) >= 3
		providedPath = char(string(varargin{3}));
	end

	store = loadStore(projectRoot);

	% If caller explicitly provided a path, use it directly with no prompt.
	if ~isempty(strtrim(providedPath))
		resolvedPath = makeAbsolutePath(providedPath, projectRoot);
		store.(key) = resolvedPath;
		saveStore(projectRoot, store);
		return;
	end

	% Determine candidate from stored value or default.
	candidate = defaultPath;
	if isfield(store, key)
		storedPath = char(string(store.(key)));
		if ~isempty(strtrim(storedPath)) && isfolder(storedPath)
			candidate = storedPath;
		end
	end

	% Prompt user to confirm or enter a different folder.
	displayPath = toRelativePath(candidate, projectRoot);
	switch key
		case 'trainRoot'
			fprintf('loading training data from: %s\n', displayPath);
		case 'testRoot'
			fprintf('Running inference test on: %s\n', displayPath);
		otherwise
			fprintf('%s: %s\n', key, displayPath);
	end
	userInput = strtrim(input('? ', 's'));
	if ~isempty(userInput)
		resolvedPath = makeAbsolutePath(userInput, projectRoot);
	else
		resolvedPath = candidate;
	end

	store.(key) = resolvedPath;
	saveStore(projectRoot, store);
end

function defaultPath = getDefaultForKey(projectRoot, key)
	switch key
		case 'trainRoot'
			defaultPath = fullfile(projectRoot, 'MNIST_digits');
		case 'testRoot'
			defaultPath = fullfile(projectRoot, 'KMNIST_japanese');
		otherwise
			error('getSetFolderPaths:unknownKey', ...
				'Unknown folder key: %s. Supported keys: trainRoot, testRoot.', key);
	end
end

function store = loadStore(projectRoot)
	store = struct();
	storeFile = getStoreFile(projectRoot);
	if ~isfile(storeFile)
		return;
	end

	data = load(storeFile, 'folderPaths');
	if isfield(data, 'folderPaths') && isstruct(data.folderPaths)
		store = data.folderPaths;
	end
end

function saveStore(projectRoot, store)
	storeFile = getStoreFile(projectRoot);
	folderPaths = store; %#ok<NASGU>
	save(storeFile, 'folderPaths', '-v7.3');
end

function storeFile = getStoreFile(projectRoot)
	cacheDir = fullfile(projectRoot, 'trained_models');
	if ~isfolder(cacheDir)
		mkdir(cacheDir);
	end
	storeFile = fullfile(cacheDir, 'folder_paths_cache.mat');
end

function absPath = makeAbsolutePath(pathInput, projectRoot)
	pathInput = strtrim(char(string(pathInput)));
	if isempty(pathInput)
		absPath = pathInput;
		return;
	end

	if isAbsolutePath(pathInput)
		absPath = pathInput;
		return;
	end

	absPath = fullfile(projectRoot, pathInput);
end

function tf = isAbsolutePath(pathInput)
	if isempty(pathInput)
		tf = false;
		return;
	end

	if startsWith(pathInput, '\\')
		tf = true;
		return;
	end

	tf = numel(pathInput) >= 2 && pathInput(2) == ':';
end

function relPath = toRelativePath(fullPath, projectRoot)
	fullPath = string(fullPath);
	projectRoot = string(projectRoot);

	if startsWith(fullPath, projectRoot)
		relPath = extractAfter(fullPath, strlength(projectRoot));
		if startsWith(relPath, filesep)
			relPath = extractAfter(relPath, 1);
		end
		relPath = "." + filesep + relPath;
	else
		relPath = fullPath;
	end

	relPath = char(relPath);
end
