function net = trainCNN(XTrain, YTrain, opts)
% trainCNN  Train a CNN for MNIST classification with latent layer

    % ------------------------------------------------------------------
    % 0. Defaults
    % ------------------------------------------------------------------
    if nargin < 3, opts = struct(); end
    latentDim  = getOpt(opts, 'latentDim', 128);
    maxEpochs  = getOpt(opts, 'maxEpochs', 20);
    miniBatch  = getOpt(opts, 'miniBatch', 256);
    learnRate  = getOpt(opts, 'learnRate', 1e-3);
    valFrac    = getOpt(opts, 'valFrac', 0.1);
    verbose    = getOpt(opts, 'verbose', true);

    nClasses = numel(categories(YTrain));

    % ------------------------------------------------------------------
    % 1. Reshape input (IMPORTANT)
    % ------------------------------------------------------------------
    XTrain = reshape(XTrain', 28, 28, 1, []);

    % ------------------------------------------------------------------
    % 2. Validation split
    % ------------------------------------------------------------------
    nTrain = size(XTrain, 4);
    nVal   = round(valFrac * nTrain);
    idx    = randperm(nTrain);

    valIdx = idx(1:nVal);
    trIdx  = idx(nVal+1:end);

    XTr = XTrain(:, :, :, trIdx);   YTr = YTrain(trIdx);
    XVl = XTrain(:, :, :, valIdx);  YVl = YTrain(valIdx);

    % ------------------------------------------------------------------
    % 3. CNN architecture
    % ------------------------------------------------------------------
    layers = [
        imageInputLayer([28 28 1], 'Name', 'input', 'Normalization', 'none')

        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

        % Flatten happens automatically before FC

        fullyConnectedLayer(latentDim, 'Name', 'fc_latent')
        reluLayer('Name', 'relu_latent')

        fullyConnectedLayer(nClasses, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    % ------------------------------------------------------------------
    % 4. Training options
    % ------------------------------------------------------------------
    trainOpts = trainingOptions('adam', ...
        'MaxEpochs',          maxEpochs, ...
        'MiniBatchSize',      miniBatch, ...
        'InitialLearnRate',   learnRate, ...
        'Shuffle',            'every-epoch', ...
        'ValidationData',     {XVl, YVl}, ...
        'ValidationFrequency', 50, ...
        'Verbose',            verbose, ...
        'Plots',              'none');

    % ------------------------------------------------------------------
    % 5. Train
    % ------------------------------------------------------------------
    net = trainNetwork(XTr, YTr, layers, trainOpts);

    % ------------------------------------------------------------------
    % 6. Validation accuracy
    % ------------------------------------------------------------------
    YPred = classify(net, XVl, 'MiniBatchSize', miniBatch);
    acc   = mean(YPred == YVl) * 100;
    fprintf('[trainCNN] Validation accuracy: %.2f%%\n', acc);
end


function v = getOpt(s, field, default)
    if isfield(s, field)
        v = s.(field);
    else
        v = default;
    end
end