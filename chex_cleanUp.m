function chex_cleanUp()
% chex_cleanUp  Remove cached model files for the CheXpert pipeline.
%
%   Deletes Stage 1, Stage 2 (CNN_chex/MLP_chex), and Stage 3 cache files
%   for the CheXpert pipeline.  Run before a forced retrain to start fresh.
%   For the MNIST digits pipeline use digits_cleanUp.

	here        = fileparts(mfilename('fullpath'));
	cacheFolder = fullfile(here, 'trained_models');

	if ~isfolder(cacheFolder)
		fprintf('Cache folder not found: %s\n', cacheFolder);
		return;
	end

	cacheFiles = { ...
		'cnn_chex_cache.mat', ...
		'mlp_chex_cache.mat', ...
		'md_chex_cache.mat', ...
		'stage3_chex_CNN_chex_16_32_64_128_fc256_lhl.mat', ...
		'stage3_chex_CNN_chex_16_32_64_128_fc256_fusion.mat', ...
		'stage3_chex_CNN_chex_16_32_64_128_fc256_mbm.mat', ...
		'stage3_chex_MLP_chex_1024_512_256_128_lhl.mat', ...
		'stage3_chex_MLP_chex_1024_512_256_128_fusion.mat', ...
		'stage3_chex_MLP_chex_1024_512_256_128_mbm.mat'};

	fprintf('=== chex_cleanUp: CheXpert pipeline ===\n');
	deletedCount = 0;
	for i = 1:numel(cacheFiles)
		f = fullfile(cacheFolder, cacheFiles{i});
		if isfile(f)
			try
				delete(f);
				fprintf('  Deleted : %s\n', cacheFiles{i});
				deletedCount = deletedCount + 1;
			catch ME
				fprintf('  Error   : %s — %s\n', cacheFiles{i}, ME.message);
			end
		else
			fprintf('  Missing : %s\n', cacheFiles{i});
		end
	end
	fprintf('Done. %d file(s) deleted.\n', deletedCount);
end
