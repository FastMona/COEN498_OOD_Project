function accuracy = testMLP(csv_file, W, b)

%% ================= LOAD DATA =================
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
