function Chex_cleanUp()
% Chex_cleanUp  Remove all CheXpert pipeline cache files.
%
%   Deletes cached models and manifolds for the CheXpert pipeline:
%     - CNN_chex network cache
%     - MLP_chex network cache
%     - MD1_chex pixel-space manifold cache
%     - MD_chex / MD2_chex latent manifold cache (legacy)
%     - MD3_chex Stage-3 manifold caches (all 3 algorithms × 2 networks)

	here        = fileparts(mfilename('fullpath'));   % .../Utilities/
	projectRoot = fileparts(here);                    % .../COEN498_OOD_Project/
	cacheFolder = fullfile(projectRoot, 'trained_models');

	if ~isfolder(cacheFolder)
		fprintf('Cache folder not found: %s\n', cacheFolder);
		return;
	end

	cacheFiles = {
		'cnn_chex_cache.mat', ...
		'mlp_chex_cache.mat', ...
		'md_chex_cache.mat', ...
		'md_chex_latent_cache.mat', ...
		'stage3_chex_CNN_chex_16_32_64_128_fc256_lhl.mat', ...
		'stage3_chex_CNN_chex_16_32_64_128_fc256_fusion.mat', ...
		'stage3_chex_CNN_chex_16_32_64_128_fc256_mbm.mat', ...
		'stage3_chex_MLP_chex_1024_512_256_128_lhl.mat', ...
		'stage3_chex_MLP_chex_1024_512_256_128_fusion.mat', ...
		'stage3_chex_MLP_chex_1024_512_256_128_mbm.mat'
	};

	fprintf('=== Chex_cleanUp: removing CheXpert cache files ===\n');

	deletedCount = 0;
	for i = 1:numel(cacheFiles)
		cacheFile = fullfile(cacheFolder, cacheFiles{i});
		if isfile(cacheFile)
			try
				delete(cacheFile);
				fprintf('Deleted: %s\n', cacheFiles{i});
				deletedCount = deletedCount + 1;
			catch ME
				fprintf('Error deleting %s: %s\n', cacheFiles{i}, ME.message);
			end
		else
			fprintf('Not found (skipping): %s\n', cacheFiles{i});
		end
	end

	fprintf('\nChex_cleanUp complete. Deleted %d file(s).\n', deletedCount);
end
