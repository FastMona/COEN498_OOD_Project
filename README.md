# Handwritten Digit OOD Filtering and Classification

Out-of-distribution (OOD) detection and classification pipeline for handwritten digit images stored in IDX format, implemented in MATLAB.

The pipeline applies three sequential stages: a pixel-space manifold gate, two independent digit classifiers, and a latent-space Mahalanobis distance post-filter. `Folder_testor` ties all three stages together.

---

## Repository Layout

```text
COEN498_OOD_Project/
    Folder_testor.m
    MD_Stage1_Prefilter.m
    MD_Stage3_Postfilter.m
    CNN_reader.m
    MLP_reader.m
    digit_eval_metrics.m
    digits_cleanUp.m
    visualize_examples.m
    visualize_ood_examples.m
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
        stage3_CNN_8_16_32_fc64_lhl.mat
        stage3_CNN_8_16_32_fc64_fusion.mat
        stage3_CNN_8_16_32_fc64_mbm.mat
        stage3_MLP_512_256_128_lhl.mat
        stage3_MLP_512_256_128_fusion.mat
        stage3_MLP_512_256_128_mbm.mat
        folder_paths_cache.mat
```

---

## Requirements

- MATLAB R2021b or later
- Deep Learning Toolbox (required for `trainNetwork`, `classify`, `confusionchart`)
- Image Processing Toolbox (recommended for `imshow` in visualization utilities)

No Python environment is needed.

---

## Datasets

| Folder | Role | Format | Size |
| --- | --- | --- | --- |
| `MNIST_digits/` | Training + in-distribution test | IDX | 60 000 train / 10 000 test, 28×28 grayscale |
| `KMNIST_japanese/` | Out-of-distribution test | IDX | 10 000 images, 28×28 grayscale |
| `MNIST_fashion/` | Alternative OOD or cross-domain test | IDX | 10 000 images, 28×28 grayscale |

IDX files use big-endian byte order. Image magic number: `0x00000803`. Label magic number: `0x00000801`.

---

## Pipeline Overview

The three stages operate sequentially. A sample must pass Stage 1 before reaching Stage 2; only Stage-1-accepted samples are scored by Stage 3.

```text
Input images
    │
    ▼
Stage 1 — Pixel-Space Pre-Filter  (MD_Stage1_Prefilter)
    │  Builds a class-conditional PCA manifold in raw pixel space.
    │  Samples whose Mahalanobis confidence < vigilance are rejected.
    │  Default: DORMANT (vigilance = 0, all samples pass).
    │
    ▼  accepted samples only
Stage 2 — Classification  (CNN_reader + MLP_reader)
    │  Two independent classifiers assign digit labels 0–9.
    │  CNN: two conv blocks + FC(64) + softmax.
    │  MLP: 784 → 512 → 256 → 128 → 10 + softmax.
    │  Agreement rate and digit distribution are reported.
    │
    ▼  accepted samples only
Stage 3 — Latent-Space Post-Filter  (MD_Stage3_Postfilter × 2 networks × 3 algorithms)
    │  Mahalanobis distance is computed in hidden-layer feature space.
    │  Three independent algorithms per network:
    │    Algorithm 1  LHL    — last hidden layer (relu4/CNN, relu3/MLP)
    │    Algorithm 2  FUSION — all layers z-normalised and concatenated
    │    Algorithm 3  MBM    — independent detector per architectural branch
    ▼
    6 Stage-3 accept/reject decisions  (CNN×3 + MLP×3)
```

Stage 1 is dormant by default for digit experiments because pixel-space MD is a weak discriminator for near-OOD patterns such as Kuzushiji digits. Set `stage1Active = true` to enable it.

---

## Quick Start

Set the MATLAB current folder to the repository root before running any script.

### Full pipeline

```matlab
results = Folder_testor();
```

Prompts for training and test folder paths on first run, then caches them. Defaults: `MNIST_digits/` (train), `KMNIST_japanese/` (test), vigilance `0.5`, Stage 1 dormant.

### Explicit arguments

```matlab
results = Folder_testor('MNIST_digits', 'KMNIST_japanese', 0.5);
results = Folder_testor('MNIST_digits', 'KMNIST_japanese', 0.5, true);  % Stage 1 active
```

### Stage 1 pre-filter only

```matlab
s1 = MD_Stage1_Prefilter('KMNIST_japanese', 0.5);
```

### Individual classifiers

```matlab
cnn = CNN_reader('MNIST_digits');
mlp = MLP_reader('MNIST_digits');
```

### Evaluation metrics (AUROC / AUPR / FPR@95TPR)

```matlab
digit_eval_metrics();   % prompts for normal and OOD folders
```

### Visualization

```matlab
visualize_examples('MNIST_digits');
visualize_ood_examples('KMNIST_japanese', 0.5, 5);
```

### Clear all digit caches

```matlab
digits_cleanUp();
```

---

## Scripts

### `MD_Stage1_Prefilter.m` — Pixel-Space Manifold Gate

Trains one PCA manifold per digit class (0–9) on the MNIST training set and scores each test sample against all ten manifolds.

**Algorithm:**

For each digit class `d`, training images are mean-centred and decomposed via SVD to retain 20 principal components:

```text
proj     = (x − μ_d) · basis_d          % project onto manifold
recon    = proj · basis_d^T             % reconstruct
res      = (x − μ_d) − recon           % off-manifold residual
distance = sqrt( sum(proj² / latent) / k  +  sum(res²) / residualVar )
```

Confidence is `exp(−0.5 · distance²) / distanceScale`, where `distanceScale` is the 95th-percentile training distance. A sample is accepted if `max(confidence across all classes) ≥ vigilance`.

**Accepted input formats:**

| Shape | Meaning |
| --- | --- |
| Folder path (string) | Reads IDX `t10k-*` files from that folder |
| `28×28` | Single image |
| `N×784` | N flattened images |
| `28×28×N` | N grayscale images |
| `28×28×1×N` | MATLAB 4-D image batch |

**Vigilance behaviour:**

| Value | Effect |
| --- | --- |
| `0.0` | Accept everything — no filtering |
| `0.5` | Moderate — recommended default |
| `1.0` | Maximum strictness — rejects nearly all input |

**Call forms:**

```matlab
s1 = MD_Stage1_Prefilter(testInput);
s1 = MD_Stage1_Prefilter(testInput, 0.7);
s1 = MD_Stage1_Prefilter(testInput, 0.5, true, false);  % force retrain, silent
```

**Cache:** `trained_models/md_filter_cache.mat`

---

### `CNN_reader.m` — Convolutional Classifier

Two-block CNN trained on the MNIST training set.

**Architecture:**

```text
Input [28, 28, 1]
  Conv2D(3×3, 8 filters) + BatchNorm + ReLU (relu1) + MaxPool(2×2)  → [14, 14, 8]
  Conv2D(3×3, 16 filters) + BatchNorm + ReLU (relu2) + MaxPool(2×2) → [7, 7, 16]
  Flatten
  FC(64) + ReLU (relu3)
  FC(10) + Softmax → digit label (0–9)
```

Training: Adam, learning rate `0.001`, 6 epochs, batch size 128. Detects `fashion` in the folder name and switches to clothing class labels automatically.

**Call forms:**

```matlab
cnn = CNN_reader('MNIST_digits');
cnn = CNN_reader('MNIST_digits', true);   % force retrain
```

**Cache:** `trained_models/cnn_reader_cache.mat`

---

### `MLP_reader.m` — Fully-Connected Classifier

Flattens each 28×28 image to a 784-dim vector and passes it through configurable hidden layers.

**Default architecture:**

```text
Input 784
  FC(512) + ReLU (relu1)
  FC(256) + ReLU (relu2)
  FC(128) + ReLU (relu3)
  FC(10) + Softmax → digit label (0–9)
```

Hidden layer widths are set via `hiddenLayerSizes` inside the file (up to 5 layers, 1–1024 neurons each). A change in architecture automatically invalidates the cache.

Training: Adam, learning rate `0.001`, 10 epochs, batch size 256.

**Call forms:**

```matlab
mlp = MLP_reader('MNIST_digits');
mlp = MLP_reader('MNIST_digits', true);   % force retrain
```

**Cache:** `trained_models/mlp_reader_cache.mat`

---

### `MD_Stage3_Postfilter.m` — Latent-Space OOD Detector

Fits a Mahalanobis distance model in the hidden-layer feature space of a trained network and scores test samples against it. The OOD threshold is the 97.5th percentile of training-set distances — no held-out OOD data is required.

Three algorithms are supported, selected via the `algo` argument:

| Algorithm | Feature source | Detector |
| --- | --- | --- |
| `LHL` | Last hidden layer only (relu4/CNN, relu3/MLP) | Single global covariance |
| `FUSION` | All hidden layers, z-normalised per layer, concatenated | Single global covariance on combined vector |
| `MBM` | Each architectural branch independently | One detector per branch; OOD if any fires |

**CNN branch layout (MBM):** `{relu1}` \| `{relu2}` \| `{relu3, relu4}` (pooling boundaries)

**MLP branch layout (MBM):** `{relu1}` \| `{relu2}` \| `{relu3}` (one branch per layer)

**Call forms:**

```matlab
% Training mode — fit model on in-distribution features
model = MD_Stage3_Postfilter('train', net, networkID, algo, layerConfig);

% Test mode — score new samples
s3 = MD_Stage3_Postfilter('test', net, testData, model);
```

**Cache:** `trained_models/stage3_<networkID>_<algo>.mat`

---

### `Folder_testor.m` — End-to-End Orchestrator

Runs the complete three-stage pipeline on a test folder.

**Workflow:**

1. Prompt for training and test folder paths (cached after first run).
2. Run `MD_Stage1_Prefilter` on test images — dormant by default.
3. Run `CNN_reader` and `MLP_reader` on Stage-1-accepted samples.
4. Run `MD_Stage3_Postfilter` under all three algorithms for both networks.
5. Report per-stage counts and a final summary table.

**Call forms:**

```matlab
results = Folder_testor();
results = Folder_testor(trainFolder, testFolder);
results = Folder_testor(trainFolder, testFolder, rejectThreshold);
results = Folder_testor(trainFolder, testFolder, rejectThreshold, stage1Active);
```

---

### `digit_eval_metrics.m` — OOD Evaluation Metrics

Scores a normal folder and an OOD folder through the pipeline and computes four detection metrics: AUROC, AUPR-Out, AUPR-In, and FPR@95TPR. Generates ROC and PR curve figures.

```matlab
digit_eval_metrics();   % interactive folder prompts
```

---

### Utilities

| Script | Purpose |
| --- | --- |
| `visualize_examples.m` | Display up to 10 random images from a folder with labels |
| `visualize_ood_examples.m` | Show accepted (top row) vs rejected (bottom row) after Stage 1 |
| `digits_cleanUp.m` | Delete all digit pipeline `.mat` caches under `trained_models/` |

---

## Output Structures

### `Folder_testor` output

| Field | Description |
| --- | --- |
| `TrainDataRoot` | Training folder path used |
| `OODDataRoot` | Test folder path used |
| `RejectThreshold` | Stage 1 vigilance value |
| `Stage1` | Full `MD_Stage1_Prefilter` result struct |
| `CNN.YPred` | CNN predicted labels for Stage-1-accepted samples |
| `MLP.YPred` | MLP predicted labels for Stage-1-accepted samples |
| `Stage3.CNN.LHL` / `FUSION` / `MBM` | Stage 3 results per algorithm (CNN) |
| `Stage3.MLP.LHL` / `FUSION` / `MBM` | Stage 3 results per algorithm (MLP) |
| `AgreementAcceptedPct` | Percentage where CNN and MLP agree on accepted samples |
| `DigitDistribution` | Table of predicted digit counts across accepted samples |
| `AcceptedMask` | Logical mask from Stage 1 |
| `MDFilter` | Alias for `Stage1` (legacy compatibility) |

### `MD_Stage1_Prefilter` output

| Field | Description |
| --- | --- |
| `BestDigit` | Nearest class manifold for each sample |
| `BestDistance` | Distance to that manifold |
| `Confidence` | Confidence score in `[0, 1]` |
| `Accepted` / `IsOOD` | Logical masks |
| `AcceptedCount` / `RejectedCount` | Scalar counts |
| `DistanceMatrix` | N×10 distances to all class manifolds |
| `ConfidenceMatrix` | N×10 confidences for all class manifolds |
| `Images4D` | Input images as 28×28×1×N array |
| `Features` | Flattened N×784 feature matrix |
| `InputLabels` | Ground-truth labels from the IDX label file |

### `CNN_reader` / `MLP_reader` output

| Field | Description |
| --- | --- |
| `Network` | Trained MATLAB `SeriesNetwork` object |
| `YTrue` | Categorical test labels |
| `YPred` | Categorical predicted labels |
| `Accuracy` | Test accuracy in percent |
| `ConfusionMatrix` | 10×10 numeric confusion matrix |
| `ConfusionTable` | Table version of the confusion matrix |

---

## Caching and Reproducibility

Each model is cached in `trained_models/` and reloaded automatically when the training data root (and for MLP, the architecture) match the cached run.

To force retraining, pass `true` as the second argument to `CNN_reader` or `MLP_reader`, or `true` as the third argument to `MD_Stage1_Prefilter`.

To remove all digit caches at once:

```matlab
digits_cleanUp();
```

---

## Troubleshooting

### Missing IDX file errors

Each dataset folder must contain files with these exact names:

```text
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

Raise the vigilance threshold, or verify the test folder contains images genuinely different from MNIST digits.

### Stage 3 cache mismatch

If the network was retrained, delete the corresponding `stage3_*.mat` files from `trained_models/` or run `digits_cleanUp()`.
