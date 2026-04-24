function digits_cleanUp()
% digits_cleanUp  Remove cached model files for the MNIST digits pipeline.
%
%   Deletes Stage 1, Stage 2 (CNN/MLP), and Stage 3 cache files for the
%   digits pipeline.  Run before a forced retrain to start fresh.
%   For the CheXpert pipeline use chex_cleanUp.

	here        = fileparts(mfilename('fullpath'));
	cacheFolder = fullfile(here, 'trained_models');

	if ~isfolder(cacheFolder)
		fprintf('Cache folder not found: %s\n', cacheFolder);
		return;
	end

	cacheFiles = { ...
		'cnn_reader_cache.mat', ...
		'mlp_reader_cache.mat', ...
		'md_filter_cache.mat', ...
		'stage3_CNN_8_16_32_fc64_lhl.mat', ...
		'stage3_CNN_8_16_32_fc64_fusion.mat', ...
		'stage3_CNN_8_16_32_fc64_mbm.mat', ...
		'stage3_MLP_512_256_128_lhl.mat', ...
		'stage3_MLP_512_256_128_fusion.mat', ...
		'stage3_MLP_512_256_128_mbm.mat'};

	fprintf('=== digits_cleanUp: digits pipeline ===\n');
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
