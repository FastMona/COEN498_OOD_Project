function relPath = getSetFolderPaths(fullPath)
% getSetFolderPaths Convert a full file path to a path relative to the project root.
%
%   relPath = getSetFolderPaths(fullPath) returns the path relative to the
%   project folder. If the path is not within the project, returns the
%   full path unchanged.
%
%   Example:
%     If project root is C:\Users\David\...\COEN498_OOD_Project and
%     fullPath is C:\Users\David\...\COEN498_OOD_Project\MNIST_digits\raw,
%     then relPath will be .\MNIST_digits\raw

	% Get the project root (folder where the main scripts are)
	projectRoot = fileparts(mfilename('fullpath'));
	
	% Normalize paths to handle different separators
	fullPath = string(fullPath);
	projectRoot = string(projectRoot);
	
	% Check if fullPath is within projectRoot
	if startsWith(fullPath, projectRoot)
		% Remove project root from path
		relPath = extractAfter(fullPath, strlength(projectRoot));
		% Remove leading separator if present
		if startsWith(relPath, filesep)
			relPath = extractAfter(relPath, 1);
		end
		% Prepend .\ to indicate current folder
		relPath = "." + filesep + relPath;
	else
		% Path is outside project, return as-is
		relPath = fullPath;
	end
	
	% Convert back to char if input was char
	relPath = char(relPath);
end
