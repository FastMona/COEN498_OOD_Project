function Z = getLatentFeatures(csv_file, W, b, num_samples)

%% ================= LOAD DATA =================
data = readmatrix(csv_file);

Y = data(:,1);        % labels (0–9)
X = data(:,2:end);    % pixels

%% ================= PREPROCESS =================
X = X / 255;
X = X';

Y = Y + 1;

% Limit samples (for speed & clarity)
if nargin > 3
    X = X(:,1:num_samples);
    Y = Y(1:num_samples);
end

N = length(Y);

%% ================= FEATURE EXTRACTION =================
L = length(W);

Z = zeros(size(W{L-1},1), N); % dimension of last hidden layer

for i = 1:N
    
    x = X(:,i);
    h = x;
    
    for l = 1:L-1   % stop before output layer
        a = W{l} * h + b{l};
        h = max(0, a);
    end
    
    Z(:,i) = h;
end

%% ================= PCA FOR VISUALIZATION =================
% Convert to samples × features
Z_t = Z';

% Center data
Z_mean = mean(Z_t,1);
Z_centered = Z_t - Z_mean;

% Covariance
C = cov(Z_centered);

% Eigen decomposition
[V, D] = eig(C);

% Sort eigenvalues descending
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% Take top 2 components
Z_2D = Z_centered * V(:,1:2);

%% ================= PLOT =================
figure;
hold on;

colors = lines(10);

for c = 1:10
    idx_c = (Y == c);
    scatter(Z_2D(idx_c,1), Z_2D(idx_c,2), 15, colors(c,:), 'filled');
end

title('Latent Feature Clusters (PCA)');
xlabel('PC1');
ylabel('PC2');
grid on;
legend('0','1','2','3','4','5','6','7','8','9');

hold off;

end