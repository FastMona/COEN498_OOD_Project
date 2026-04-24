function cleanUp()
% cleanUp Remove temporary cache files for trained models and manifolds.
%
%   cleanUp() deletes the cached training files:
%     - CNN model cache
%     - MLP model cache
%     - Manifold model cache
%     - Stored folder path cache
%
%   This function is useful for clearing cached models before retraining
%   or when running a fresh training session.

	here = fileparts(mfilename('fullpath'));
	cacheFolder = fullfile(here, 'trained_models');
	
	% Check if cache folder exists
	if ~isfolder(cacheFolder)
		fprintf('Cache folder not found: %s\n', getSetFolderPaths(cacheFolder));
		return;
	end
	
	% Define cache files to delete
	cacheFiles = {
		'cnn_reader_cache.mat', ...
		'mlp_reader_cache.mat', ...
		'md_filter_cache.mat', ...
		'folder_paths_cache.mat', ...
		'stage3_CNN_8_16_32_fc64_lhl.mat', ...
		'stage3_CNN_8_16_32_fc64_fusion.mat', ...
		'stage3_CNN_8_16_32_fc64_mbm.mat', ...
		'stage3_MLP_512_256_128_lhl.mat', ...
		'stage3_MLP_512_256_128_fusion.mat', ...
		'stage3_MLP_512_256_128_mbm.mat'
	};
	
	fprintf('=== Cleaning up cache files ===\n');
	
	deletedCount = 0;
	for i = 1:numel(cacheFiles)
		cacheFile = fullfile(cacheFolder, cacheFiles{i});
		if isfile(cacheFile)
			try
				delete(cacheFile);
				fprintf('Deleted: %s\n', getSetFolderPaths(cacheFile));
				deletedCount = deletedCount + 1;
			catch ME
				fprintf('Error deleting %s: %s\n', cacheFiles{i}, ME.message);
			end
		else
			fprintf('Not found: %s\n', getSetFolderPaths(cacheFile));
		end
	end
	
	fprintf('\nCleanup complete. Deleted %d file(s).\n', deletedCount);
end
