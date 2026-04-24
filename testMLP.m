function accuracy = testMLP(csv_file, W, b)

%% ================= ARCHITECTURE =================
hiddenSizes = [512, 256, 128];
inputSize   = 784;
outputSize  = 10;

%% ================= DEFAULTS =================
if nargin < 1 || isempty(csv_file)
    csv_file = 'mnist_test.csv';
end

%% ================= INITIALIZE W AND b IF NOT PROVIDED =================
if nargin < 2 || isempty(W)
    sizes = [inputSize, hiddenSizes, outputSize];
    L_init = numel(sizes) - 1;
    W = cell(L_init, 1);
    b = cell(L_init, 1);
    for l = 1:L_init
        fanIn  = sizes(l);
        W{l} = randn(sizes(l+1), fanIn) * sqrt(2 / fanIn);
        b{l} = zeros(sizes(l+1), 1);
    end
    fprintf("W and b initialised with He init (untrained — expect ~10%% accuracy)\n");
end

%% ================= LOAD DATA =================
if isfolder(csv_file)
    [X, Y] = loadMNISTForMLP(csv_file);   % 784×N, labels 1–10
    N = length(Y);
else
    data = readmatrix(csv_file);
    Y = data(:,1);        % labels (0–9)
    X = data(:,2:end);    % pixels

    %% ================= PREPROCESS =================

    % Normalize
    X = X / 255;

    % Transpose → (features × samples)
    X = X';

    % Fix labels (0→9 → 1→10)
    Y = Y + 1;

    N = length(Y);
end

%% ================= TEST =================
correct = 0;
L = length(W);

for i = 1:N
    
    x = X(:,i);
    h = x;
    
    %% -------- FORWARD --------
    for l = 1:L
        a = W{l} * h + b{l};
        
        if l == L
            h = softmax(a);
        else
            h = max(0, a);
        end
    end
    
    %% -------- PREDICTION --------
    [~, pred] = max(h);
    
    if pred == Y(i)
        correct = correct + 1;
    end
end

accuracy = correct / N;

fprintf("Test Accuracy: %.2f%%\n", accuracy * 100);

end
