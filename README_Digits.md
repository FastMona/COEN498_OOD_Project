# Handwritten Digit OOD Filtering and Classification

Out-of-distribution (OOD) detection and classification pipeline for handwritten digit images stored in IDX format, implemented in MATLAB.

The pipeline combines a manifold-distance OOD filter with two independent classifiers (CNN and MLP). An end-to-end orchestrator first rejects samples that do not resemble any known digit, then runs both classifiers on the remaining accepted samples.

---

## Repository Layout

```
COEN498_OOD_Project/
    MD_filter.m
    CNN_reader.m
    MLP_reader.m
    Folder_testor.m
    getSetFolderPaths.m
    Utilities/
        visualize_examples.m
        visualize_ood_examples.m
        cleanUp.m
    MNIST_digits/
        train-images-idx3-ubyte
        train-labels-idx1-ubyte
        t10k-images-idx3-ubyte
        t10k-labels-idx1-ubyte
    KMNIST_japanese/
        t10k-images-idx3-ubyte
        t10k-labels-idx1-ubyte
    trained_models/
        md_filter_cache.mat
        cnn_reader_cache.mat
        mlp_reader_cache.mat
        folder_paths_cache.mat
```

The dataset folders already follow the naming conventions expected by the scripts.

---

## Requirements

- MATLAB R2021b or later
- Deep Learning Toolbox (required for `trainNetwork`, `classify`, `confusionchart`)
- Image Processing Toolbox (recommended for `imshow` in visualization utilities)

No Python environment is needed.

---

## Datasets

| Folder | Role | Format | Size |
|---|---|---|---|
| `MNIST_digits/` | Training + in-distribution test | IDX | 60 000 train / 10 000 test, 28×28 grayscale |
| `KMNIST_japanese/` | Out-of-distribution test | IDX | 10 000 images, 28×28 grayscale |
| `MNIST_fashion/` | Alternative OOD or cross-domain test | IDX | 10 000 images, 28×28 grayscale |

IDX files use big-endian byte order. Image magic number: `0x00000803`. Label magic number: `0x00000801`.

---

## Quick Start

Set the MATLAB current folder to the repository root before running any script.

### Full pipeline

```matlab
results = Folder_testor();
```

Defaults: training root `MNIST_digits/`, OOD test folder `KMNIST_japanese/`, vigilance threshold `0.5`. Saved paths from `trained_models/folder_paths_cache.mat` are used when available.

### Explicit arguments

```matlab
results = Folder_testor('MNIST_digits/', 'KMNIST_japanese/', 0.5);
```

### OOD filter only

```matlab
md = MD_filter('KMNIST_japanese/', 0.5);
```

### Individual classifiers

```matlab
cnn = CNN_reader('MNIST_digits/');
mlp = MLP_reader('MNIST_digits/');
```

### Visualization

```matlab
visualize_examples('MNIST_digits/');
visualize_ood_examples('KMNIST_japanese/', 0.5, 5);
```

### Clear all caches

```matlab
cleanUp();
```

---

## Scripts

### `MD_filter.m` — Manifold-Distance OOD Detector

Trains one PCA manifold per digit class (0–9) on the MNIST training set and scores each test sample against all ten manifolds.

**Algorithm:**

For each digit class `d`, the training images of that class are mean-centred and decomposed via SVD to retain 20 principal components. Two quantities are stored:

- `latent` — variance along each principal direction
- `residualVar` — median residual energy from PCA reconstruction

At inference, the distance of a test image `x` to class `d` is:

```
proj     = (x − μ_d) · basis_d          % project onto manifold
recon    = proj · basis_d^T             % reconstruct
res      = (x − μ_d) − recon           % residual (off-manifold component)
distance = sqrt( sum(proj² / latent) / k  +  sum(res²) / residualVar )
```

The per-class confidence is `exp(−0.5 · distance²) / distanceScale`, where `distanceScale` is the 95th-percentile training distance. A sample is accepted if `max(confidence across all digits) ≥ vigilance`.

**Accepted input formats:**

| Shape | Meaning |
|---|---|
| Folder path (string) | Reads IDX `t10k-*` files from that folder |
| `28×28` | Single image |
| `N×784` | N flattened images |
| `28×28×N` | N grayscale images |
| `28×28×1×N` | MATLAB 4-D image batch |

**Vigilance (threshold) behaviour:**

| Value | Effect |
|---|---|
| `0.0` | Accept everything — no filtering |
| `0.5` | Moderate — recommended default |
| `1.0` | Maximum strictness — rejects nearly all input |

**Call forms:**

```matlab
MD_filter();                                      % train / load model only
res = MD_filter(testInput);                       % score at default vigilance 0.5
res = MD_filter(testInput, 0.7);                  % custom threshold
res = MD_filter(testInput, 0.5, true, false);     % force retrain, silent output
```

**Cache:** `trained_models/md_filter_cache.mat`

---

### `CNN_reader.m` — Convolutional Classifier

Two-block CNN trained on the MNIST training set.

**Architecture:**

```
Input [28, 28, 1]
  Conv2D(3×3, 8 filters) + BatchNorm + ReLU + MaxPool(2×2)  → [14, 14, 8]
  Conv2D(3×3, 16 filters) + BatchNorm + ReLU + MaxPool(2×2) → [7, 7, 16]
  Flatten
  FC(64) + ReLU
  FC(10) + Softmax → digit label (0–9)
```

Training: ADAM, learning rate `0.001`, 6 epochs, batch size 128. Validation on the IDX test set during training.

The script detects `fashion` in the folder name and switches to clothing labels automatically.

**Call forms:**

```matlab
cnn = CNN_reader();
cnn = CNN_reader('MNIST_digits/');
cnn = CNN_reader('MNIST_digits/', true);    % force retrain
```

**Cache:** `trained_models/cnn_reader_cache.mat`

---

### `MLP_reader.m` — Fully-Connected Classifier

Flattens each 28×28 image to a 784-dim vector and passes it through configurable hidden layers.

**Default architecture:**

```
Input 784
  FC(512) + ReLU
  FC(256) + ReLU
  FC(128) + ReLU
  FC(10) + Softmax → digit label (0–9)
```

Hidden layer widths are set via `hiddenLayerSizes` inside the file (up to 5 layers, 1–1024 neurons each). A change in architecture automatically invalidates the cache.

Training: ADAM, learning rate `0.001`, 10 epochs, batch size 256.

**Call forms:**

```matlab
mlp = MLP_reader();
mlp = MLP_reader('MNIST_digits/');
mlp = MLP_reader('MNIST_digits/', true);   % force retrain
```

**Cache:** `trained_models/mlp_reader_cache.mat`

---

### `Folder_testor.m` — End-to-End Orchestrator

Ties `MD_filter`, `CNN_reader`, and `MLP_reader` into a single evaluation run.

**Workflow:**

1. Load training data from `trainFolder` (default: `MNIST_digits/`).
2. Run `MD_filter` on test images from `testFolder` (default: `KMNIST_japanese/`).
3. Discard rejected samples.
4. Run `CNN_reader` and `MLP_reader` on accepted samples only.
5. Report CNN/MLP agreement rate and predicted digit distribution.

Console output always shows:

```
loading training data from: MNIST_digits/
Running inference test on: KMNIST_japanese/
```

**Call forms:**

```matlab
results = Folder_testor();
results = Folder_testor(trainFolder, testFolder);
results = Folder_testor(trainFolder, testFolder, 0.5);
```

`trainFolder` and `testFolder` are optional. When omitted, saved paths from `getSetFolderPaths` are used, falling back to code defaults.

---

### `getSetFolderPaths.m` — Path Resolver and Cache

Resolves the training and test folder roots. On first call it stores the paths to `trained_models/folder_paths_cache.mat` so subsequent calls do not require re-entry.

---

### Utilities

| Script | Purpose |
|---|---|
| `visualize_examples.m` | Display up to 10 random images from an IDX test set with labels |
| `visualize_ood_examples.m` | Show accepted (top row) vs rejected (bottom row) examples after OOD filtering |
| `cleanUp.m` | Delete all `.mat` cache files under `trained_models/` |

---

## Output Structures

### `Folder_testor` output

| Field | Description |
|---|---|
| `results.MDFilter` | Full OOD filter result struct (see below) |
| `results.CNN.YPred` | CNN predicted labels for accepted samples |
| `results.MLP.YPred` | MLP predicted labels for accepted samples |
| `results.AgreementAcceptedPct` | Percentage of accepted samples where CNN and MLP agree |
| `results.DigitDistribution` | Table of predicted digit counts across accepted samples |

### `MD_filter` output (score mode)

| Field | Description |
|---|---|
| `BestDigit` | Nearest digit manifold for each sample |
| `BestDistance` | Distance to that manifold |
| `Confidence` | Confidence score in `[0, 1]` |
| `Accepted` | Logical mask — true for accepted samples |
| `IsOOD` | Logical mask — true for rejected samples |
| `AcceptedCount` / `RejectedCount` | Scalar counts |
| `DistanceMatrix` | N×10 distances to all digit manifolds |
| `ConfidenceMatrix` | N×10 confidences for all digit manifolds |
| `Images4D` | Input images as 28×28×1×N array |
| `Features` | Flattened N×784 feature matrix |
| `InputLabels` | Ground-truth labels from the IDX label file |

### `CNN_reader` / `MLP_reader` output

Both return a struct with fields: `net`, `YPred`, `YTest`, `Accuracy`, `ConfusionMatrix`.

---

## Caching and Reproducibility

Each model is cached in `trained_models/` and reloaded automatically when:

- The training data root matches the cached path.
- The MLP architecture (hidden layer sizes) matches the cached configuration.

To force retraining, pass `true` as the second argument:

```matlab
CNN_reader('MNIST_digits/', true);
MLP_reader('MNIST_digits/', true);
MD_filter('KMNIST_japanese/', 0.5, true);
```

To remove all caches at once:

```matlab
cleanUp();
```

---

## Troubleshooting

### Missing IDX file errors

Each dataset folder must contain files with these exact names:

```
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

### `trainNetwork` or `confusionchart` not found

Install or enable the MATLAB Deep Learning Toolbox.

### All OOD samples rejected

Lower the vigilance threshold:

```matlab
results = Folder_testor([], [], 0.3);
```

### All OOD samples accepted

Raise the vigilance threshold or verify that the test folder contains images genuinely different from MNIST digits.
