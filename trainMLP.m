function net = trainMLP(XTrain, YTrain, opts)
% trainMLP  Train a fully-connected MLP for MNIST classification.
%
%   net = trainMLP(XTrain, YTrain)
%   net = trainMLP(XTrain, YTrain, opts)
%
%   Architecture (inspired by Venkataramanan et al., ICCVW 2023):
%       Input(784) -> FC(512) -> BN -> ReLU ->
%                    FC(256) -> BN -> ReLU ->
%                    FC(latentDim) -> BN -> ReLU   <-- latent layer
%                    FC(nClasses) -> Softmax
%
%   The penultimate (latent) layer is what getLatentFeatures() taps into
%   for Mahalanobis distance-based OOD detection, following Lee et al.
%   (NeurIPS 2018) and the best-practice guidance in Anthony & Kamnitsas
%   (2023) that the optimal layer depends on the OOD pattern.
%
%   Inputs:
%       XTrain    - (N x 784) single matrix of training images
%       YTrain    - (N x 1)   categorical class labels
%       opts      - (optional) struct with fields:
%           .latentDim   latent layer width       (default: 128)
%           .maxEpochs   training epochs          (default: 20)
%           .miniBatch   mini-batch size          (default: 256)
%           .learnRate   initial learning rate    (default: 1e-3)
%           .valFrac     fraction held out for validation (default: 0.1)
%           .verbose     print training progress  (default: true)
%
%   Output:
%       net - trained dlnetwork object

    % ------------------------------------------------------------------
    % 0. Defaults
    % ------------------------------------------------------------------
    if nargin < 3, opts = struct(); end
    latentDim  = getOpt(opts, 'latentDim',  128);
    maxEpochs  = getOpt(opts, 'maxEpochs',  20);
    miniBatch  = getOpt(opts, 'miniBatch',  256);
    learnRate  = getOpt(opts, 'learnRate',  1e-3);
    valFrac    = getOpt(opts, 'valFrac',    0.1);
    verbose    = getOpt(opts, 'verbose',    true);

    nClasses = numel(categories(YTrain));
    fprintf('[trainMLP] Classes: %d  |  Latent dim: %d\n', nClasses, latentDim);

    % ------------------------------------------------------------------
    % 1. Validation split
    % ------------------------------------------------------------------
    nTrain = size(XTrain, 1);
    nVal   = round(valFrac * nTrain);
    idx    = randperm(nTrain);
    valIdx = idx(1:nVal);
    trIdx  = idx(nVal+1:end);

    XTr = XTrain(trIdx, :);   YTr = YTrain(trIdx);
    XVl = XTrain(valIdx, :);  YVl = YTrain(valIdx);

    % ------------------------------------------------------------------
    % 2. Build network layers
    % ------------------------------------------------------------------
    layers = [
        featureInputLayer(784, 'Name', 'input', 'Normalization', 'none')

        fullyConnectedLayer(512, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')

        fullyConnectedLayer(256, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')

        % --- Latent layer (tapped by getLatentFeatures) ---
        fullyConnectedLayer(latentDim, 'Name', 'fc_latent')
        batchNormalizationLayer('Name', 'bn_latent')
        reluLayer('Name', 'relu_latent')

        % --- Classification head ---
        fullyConnectedLayer(nClasses, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    % ------------------------------------------------------------------
    % 3. Training options
    % ------------------------------------------------------------------
    valDS = arrayDatastore([XVl, double(YVl)], 'IterationDimension', 1);

    verboseStr = 'none';
    if verbose, verboseStr = 'training-progress'; end  %#ok

    trainOpts = trainingOptions('adam', ...
        'MaxEpochs',          maxEpochs, ...
        'MiniBatchSize',      miniBatch, ...
        'InitialLearnRate',   learnRate, ...
        'LearnRateSchedule',  'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'Shuffle',            'every-epoch', ...
        'ValidationData',     {XVl, YVl}, ...
        'ValidationFrequency', 50, ...
        'Verbose',            verbose, ...
        'Plots',              'none');

    % ------------------------------------------------------------------
    % 4. Train
    % ------------------------------------------------------------------
    fprintf('[trainMLP] Training for %d epochs...\n', maxEpochs);
    net = trainNetwork(XTr, YTr, layers, trainOpts);
    fprintf('[trainMLP] Training complete.\n');

    % ------------------------------------------------------------------
    % 5. Quick accuracy report on validation set
    % ------------------------------------------------------------------
    YPred = classify(net, XVl, 'MiniBatchSize', miniBatch);
    acc   = mean(YPred == YVl) * 100;
    fprintf('[trainMLP] Validation accuracy: %.2f%%\n', acc);
end


% -------------------------------------------------------------------------
function v = getOpt(s, field, default)
    if isfield(s, field)
        v = s.(field);
    else
        v = default;
    end
end