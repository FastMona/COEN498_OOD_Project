function [XTrain, YTrain, XTest, YTest, XOod, YOod] = loadMNIST(oodDigits)


    % ==================================================================
    %  FILE PATHS  –  edit these lines to match your local setup
    % ==================================================================

    % --- In-distribution training files ---
    TRAIN_IMAGES = 'train-images-idx3-ubyte';
    TRAIN_LABELS = 'train-labels-idx1-ubyte';

    % --- In-distribution test files ---
    TEST_IMAGES  = 'test-images-idx3-ubyte';
    TEST_LABELS  = 'test-labels-idx1-ubyte';

    % --- OOD file paths (only used when USE_SEPARATE_OOD = true) ---
    OOD_IMAGES   = 'test-images-idx3(Jap)-ubyte';   % ← your OOD image file
    OOD_LABELS   = 'test-labels-idx1(Jap)-ubyte';   % ← your OOD label file

    % --- Set to true to load OOD from its own file (OPTION A) ---
    %     Set to false to carve OOD out of the test set (OPTION B) ---
    USE_SEPARATE_OOD = false;

    % ==================================================================

    if nargin < 1 || isempty(oodDigits)
        oodDigits = [8, 9];
    end

    % ------------------------------------------------------------------
    % 1. Read training files
    % ------------------------------------------------------------------
    fprintf('[loadMNIST] Reading training images : %s\n', TRAIN_IMAGES);
    XTrainRaw = readIDXImages(TRAIN_IMAGES);
    fprintf('[loadMNIST] Reading training labels : %s\n', TRAIN_LABELS);
    labTrain  = readIDXLabels(TRAIN_LABELS);

    % ------------------------------------------------------------------
    % 2. Read test files
    % ------------------------------------------------------------------
    fprintf('[loadMNIST] Reading test images     : %s\n', TEST_IMAGES);
    XTestRaw  = readIDXImages(TEST_IMAGES);
    fprintf('[loadMNIST] Reading test labels     : %s\n', TEST_LABELS);
    labTest   = readIDXLabels(TEST_LABELS);

    % ------------------------------------------------------------------
    % 3. Normalise pixels to [0, 1]
    % ------------------------------------------------------------------
    XTrainRaw = single(XTrainRaw) / 255;
    XTestRaw  = single(XTestRaw)  / 255;

    % ------------------------------------------------------------------
    % 4. OOD data
    % ------------------------------------------------------------------
    if USE_SEPARATE_OOD
        % OPTION A – load OOD from its own dedicated file
        fprintf('[loadMNIST] Reading OOD images       : %s\n', OOD_IMAGES);
        XOodRaw  = readIDXImages(OOD_IMAGES);
        fprintf('[loadMNIST] Reading OOD labels       : %s\n', OOD_LABELS);
        labOod   = readIDXLabels(OOD_LABELS);

        XOodRaw  = single(XOodRaw) / 255;

        % Training and test sets use ALL of their samples (no carve-out)
        XTrain = XTrainRaw;
        YTrain = categorical(labTrain);
        XTest  = XTestRaw;
        YTest  = categorical(labTest);
        XOod   = XOodRaw;
        YOod   = categorical(labOod);

        fprintf('[loadMNIST] OOD source: separate file (%d samples)\n', ...
                size(XOod,1));
    else
        % OPTION B – carve OOD digits out of the test set
        idMaskTrain =  ~ismember(labTrain, oodDigits);
        idMaskTest  =  ~ismember(labTest,  oodDigits);
        oodMaskTest =   ismember(labTest,  oodDigits);

        XTrain = XTrainRaw(idMaskTrain, :);
        YTrain = categorical(labTrain(idMaskTrain));
        XTest  = XTestRaw(idMaskTest,  :);
        YTest  = categorical(labTest(idMaskTest));
        XOod   = XTestRaw(oodMaskTest, :);
        YOod   = categorical(labTest(oodMaskTest));

        fprintf('[loadMNIST] OOD source: test set digits %s (%d samples)\n', ...
                num2str(oodDigits), size(XOod,1));
    end

    % ------------------------------------------------------------------
    % 5. Summary
    % ------------------------------------------------------------------
    fprintf('[loadMNIST] ID  training samples : %d\n', size(XTrain,1));
    fprintf('[loadMNIST] ID  test     samples : %d\n', size(XTest,1));
    fprintf('[loadMNIST] OOD          samples : %d\n', size(XOod,1));
end


% =========================================================================
%  Local helpers – IDX binary format readers
% =========================================================================

function X = readIDXImages(filepath)
% readIDXImages  Read an IDX3 image file and return an (N x 784) matrix.
    fid  = openIDX(filepath, 2051);
    N    = fread(fid, 1, 'uint32', 0, 'ieee-be');
    rows = fread(fid, 1, 'uint32', 0, 'ieee-be');
    cols = fread(fid, 1, 'uint32', 0, 'ieee-be');
    raw  = fread(fid, N * rows * cols, 'uint8');
    fclose(fid);
    X = reshape(raw, rows * cols, N)';   % N x 784
end


function labels = readIDXLabels(filepath)
% readIDXLabels  Read an IDX1 label file and return an (N x 1) double vector.
    fid    = openIDX(filepath, 2049);
    N      = fread(fid, 1, 'uint32', 0, 'ieee-be');
    labels = fread(fid, N, 'uint8');
    fclose(fid);
end


function fid = openIDX(filepath, expectedMagic)
% openIDX  Open an IDX file and validate its magic number.
    if ~isfile(filepath)
        error(['[loadMNIST] File not found: %s\n' ...
               'Update the FILE PATHS section at the top of loadMNIST.m.'], ...
               filepath);
    end
    fid   = fopen(filepath, 'rb', 'ieee-be');
    magic = fread(fid, 1, 'uint32', 0, 'ieee-be');
    if magic ~= expectedMagic
        fclose(fid);
        error('[loadMNIST] Unexpected magic number %d in %s (expected %d).', ...
              magic, filepath, expectedMagic);
    end
end