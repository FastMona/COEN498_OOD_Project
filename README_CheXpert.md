# Chest X-Ray Anomaly Detection — CheXpert Pipeline

A three-stage out-of-distribution (OOD) detection pipeline for chest radiographs, implemented in MATLAB. The system is trained exclusively on normal X-rays and flags abnormal images without requiring annotated abnormal examples during training.

---

## Repository Layout

```
COEN498_OOD_Project/
    MD_chex.m
    MD1_chex.m
    MD2_chex.m
    CNN_chex.m
    MLP_chex.m
    Chex_tester.m
    Utilities/
        visualize_chex_examples.m
        insert_donut_artifact.m
        insert_square_artifact.m
        pad_chex.py
        cleanUp.m
    chex_train/          (41 527 normal JPEGs, 390×320 px)
    chex_donut75/        (3 000 artificially corrupted JPEGs)
    chex_squares75/      (3 000 artificially corrupted JPEGs)
    chex_pacemaker/      (3 535 real abnormal JPEGs)
    trained_models/
        cnn_chex_cache.mat
        mlp_chex_cache.mat
        md1_chex_cache.mat
        md2_cnn_chex_cache.mat
        md2_mlp_chex_cache.mat
```

---

## Requirements

- MATLAB R2021b or later
- Deep Learning Toolbox (required for `trainNetwork`, `predict`)
- Image Processing Toolbox (required for `imread`, `imresize`, `imshow`)
- Python 3 with Pillow (required only for the `pad_chex.py` preprocessing utility)

---

## Datasets

| Folder | Contents | Role |
|---|---|---|
| `chex_train/` | 41 527 normal chest X-rays, 390×320 px, grayscale JPEG | Training data — all labelled normal |
| `chex_donut75/` | 3 000 X-rays with synthetic circular artefacts | Synthetic abnormal test set |
| `chex_squares75/` | 3 000 X-rays with synthetic rectangular artefacts | Synthetic abnormal test set |
| `chex_pacemaker/` | 3 535 X-rays with visible pacemaker hardware | Real abnormal test set |

All images are 390×320 grayscale JPEGs. Pixel values are normalised to `[0, 1]` (single precision) before being passed to any network.

### Image preparation (one-time setup)

If you have raw CheXpert images at a different resolution, use the bundled Python utility to pad them to 390×320:

```bash
python Utilities/pad_chex.py <source_folder> <output_folder>
```

### Generating synthetic artefacts

```matlab
insert_donut_artifact('<source_folder>', '<output_folder>', numImages);
insert_square_artifact('<source_folder>', '<output_folder>', numImages);
```

---

## Quick Start

Set the MATLAB current folder to the repository root.

### Full three-stage pipeline

```matlab
results = Chex_tester('chex_donut75');      % test on donut-artefact images
results = Chex_tester('chex_squares75');    % test on square-artefact images
results = Chex_tester('chex_pacemaker');    % test on pacemaker images
```

### Run individual stages

```matlab
% Stage 1 only (pixel-space manifold prefilter)
results = MD1_chex('chex_donut75', 'chex_train', 0.5);

% Stages 1 + 2 + 3 (full MD pipeline)
results = MD_chex('chex_donut75', 'chex_train', 0.5);

% Stage 3 only (latent-space manifold filter)
results = MD2_chex('chex_donut75', 'chex_train', 0.5);
```

### Train / evaluate regression networks independently

```matlab
cnn = CNN_chex('chex_train');
mlp = MLP_chex('chex_train');
```

### Visualise examples

```matlab
visualize_chex_examples('chex_pacemaker', 8);
```

### Clear all caches

```matlab
cleanUp();
```

---

## Pipeline Overview

The pipeline is designed around the assumption that only normal images are available at training time. Abnormality is detected as deviation from the learned normal distribution.

```
Test image
    │
    ▼
Stage 1 — Pixel-space manifold distance (MD1_chex)
    │  Reject if far from normal pixel manifold
    ▼
Stage 2 — Regression scoring (CNN_chex + MLP_chex)
    │  Score ≈ 1.0 → normal,  Score ≪ 1.0 → abnormal
    ▼
Stage 3 — Latent-space manifold distance (MD2_chex)
       Reject if far from normal latent manifold
```

Each stage can be used independently or composed via `MD_chex` / `Chex_tester`.

---

## Scripts

### `Chex_tester.m` — End-to-End Driver

Runs the full three-stage pipeline on a test folder and prints a summary report.

**Workflow:**

1. Load and preprocess all images from `testFolder`.
2. Run Stage 1 (`MD1_chex`): reject images far from the pixel-space manifold.
3. Run Stage 2 (`CNN_chex`, `MLP_chex`): compute regression anomaly scores.
4. Run Stage 3 (`MD2_chex`): reject images far from the latent-space manifold.
5. Report acceptance rates and score distributions at each stage.

**Call forms:**

```matlab
results = Chex_tester('chex_donut75');
results = Chex_tester('chex_donut75', 'chex_train');
results = Chex_tester('chex_donut75', 'chex_train', 0.5);
```

The second argument (training folder) defaults to `chex_train`. The third argument is the vigilance threshold in `[0, 1]`.

---

### `MD_chex.m` — Full Three-Stage Manifold Pipeline

Implements all three stages in one call.

**Call forms:**

```matlab
results = MD_chex('chex_donut75', 'chex_train', 0.5);
results = MD_chex('chex_donut75', 'chex_train', 0.5, true);  % force retrain
```

---

### `MD1_chex.m` — Stage 1: Pixel-Space Manifold Prefilter

Trains a single PCA manifold on the flattened pixel vectors of all normal training images (124 800 features per image, one manifold for the normal class). A test image is rejected if its manifold distance exceeds the threshold derived from the training distribution.

**Algorithm:** identical to the digit manifold (see below), applied to the full image vector rather than per-class manifolds.

**Call forms:**

```matlab
results = MD1_chex('chex_donut75', 'chex_train', 0.5);
results = MD1_chex('chex_donut75', 'chex_train', 0.5, true);  % force retrain
```

**Cache:** `trained_models/md1_chex_cache.mat`

---

### `MD2_chex.m` — Stage 3: Latent-Space Manifold Filter

Extracts penultimate-layer activations from CNN and MLP after Stage 2, then builds a PCA manifold on those latent vectors from the normal training images. Test images whose latent representations fall far from this manifold are flagged as abnormal.

- CNN latent source: `relu5` layer (FC output before final regression head)
- MLP latent source: `relu4` layer (last hidden layer)

**Call forms:**

```matlab
results = MD2_chex('chex_donut75', 'chex_train', 0.5);
results = MD2_chex('chex_donut75', 'chex_train', 0.5, true);  % force retrain
```

**Caches:** `trained_models/md2_cnn_chex_cache.mat`, `trained_models/md2_mlp_chex_cache.mat`

---

### `CNN_chex.m` — Convolutional Regression Network

Trained as a one-class regressor: all normal training images have target label `1.0`. A score close to `1.0` indicates a normal X-ray; a lower score indicates anomaly.

**Architecture:**

```
Input [320, 390, 1]
  Conv2D(3×3, 16 filters) + BatchNorm + ReLU + MaxPool(2×2)  → [160, 195, 16]
  Conv2D(3×3, 32 filters) + BatchNorm + ReLU + MaxPool(2×2)  → [ 80,  97, 32]
  Conv2D(3×3, 64 filters) + BatchNorm + ReLU + MaxPool(2×2)  → [ 40,  48, 64]
  Conv2D(3×3,128 filters) + BatchNorm + ReLU + MaxPool(2×2)  → [ 20,  24,128]
  Flatten
  FC(256) + ReLU         ← latent features extracted here (relu5)
  FC(1)  → anomaly score (regression)
```

Training: ADAM, learning rate `0.0001`, 5 epochs, batch size 32. 80/20 train/validation split (random seed 42). Early stopping if validation RMSE does not improve.

**Call forms:**

```matlab
cnn = CNN_chex('chex_train');
cnn = CNN_chex('chex_train', true);   % force retrain
```

**Cache:** `trained_models/cnn_chex_cache.mat`

---

### `MLP_chex.m` — Fully-Connected Regression Network

Flattens each 390×320 image to a 124 800-dim vector and passes it through four hidden layers. Same one-class regression objective as `CNN_chex`.

**Architecture:**

```
Input 124 800 (flattened 390×320)
  FC(1024) + ReLU
  FC(512)  + ReLU
  FC(256)  + ReLU
  FC(128)  + ReLU    ← latent features extracted here (relu4)
  FC(1)   → anomaly score (regression)
```

Training: ADAM, learning rate `0.0001`, 5 epochs, batch size 32. Same split and early-stopping logic as `CNN_chex`.

**Call forms:**

```matlab
mlp = MLP_chex('chex_train');
mlp = MLP_chex('chex_train', true);   % force retrain
```

**Cache:** `trained_models/mlp_chex_cache.mat`

---

### Utilities

| Script | Purpose |
|---|---|
| `visualize_chex_examples.m` | Display a random sample of X-ray images from a folder |
| `insert_donut_artifact.m` | Add circular artefacts to normal X-rays to create synthetic abnormal images |
| `insert_square_artifact.m` | Add rectangular artefacts to normal X-rays |
| `pad_chex.py` | Python script — pad raw CheXpert images to 390×320 with black borders |
| `cleanUp.m` | Delete all `.mat` cache files under `trained_models/` |

---

## Vigilance Threshold

The vigilance parameter (range `[0, 1]`) controls the acceptance boundary for the manifold-distance stages.

| Value | Effect |
|---|---|
| `0.0` | Accept all images — no filtering |
| `0.5` | Moderate filtering — recommended default |
| `1.0` | Maximum strictness — rejects nearly all images |

The same threshold is applied to both Stage 1 (pixel manifold) and Stage 3 (latent manifold). Tune it based on the desired false-positive/false-negative trade-off for your application.

---

## Output Structures

### `Chex_tester` / `MD_chex` output

| Field | Description |
|---|---|
| `MDScores` | Stage 1 manifold distances for all test images |
| `Accepted` | Logical mask — images passing Stage 1 |
| `IsOOD` | Logical mask — images rejected by Stage 1 |
| `AcceptedCount` | Number of images passing Stage 1 |
| `CNNScores` | Stage 2 regression scores from the CNN (normal ≈ 1.0) |
| `MLPScores` | Stage 2 regression scores from the MLP |
| `CombinedScores` | Average of CNN and MLP scores |
| `LatentScores` | Stage 3 manifold distances in latent space |
| `LatentAccepted` | Logical mask — images passing Stage 3 |
| `LatentAcceptedCount` | Number of images passing Stage 3 |

### `CNN_chex` / `MLP_chex` output

| Field | Description |
|---|---|
| `net` | Trained MATLAB network object |
| `Scores` | Predicted anomaly scores on the test split |
| `RMSE` | Root mean squared error on the test split |
| `LatentFeatures` | Penultimate-layer activations for the training set |

---

## Caching and Reproducibility

All trained models are saved under `trained_models/` and reloaded on subsequent calls when the training folder path matches the cached value.

To force retraining regardless of cache:

```matlab
CNN_chex('chex_train', true);
MLP_chex('chex_train', true);
MD_chex('chex_donut75', 'chex_train', 0.5, true);
```

To remove all caches at once:

```matlab
cleanUp();
```

The train/validation split uses a fixed random seed (`42`) so repeated training runs without a cache produce the same split.

---

## Troubleshooting

### Images fail to load

Verify that all JPEGs in the dataset folders are 390×320 grayscale. Run `pad_chex.py` if images have a different resolution.

### `trainNetwork` or `predict` not found

Install or enable the MATLAB Deep Learning Toolbox.

### Stage 1 rejects all or almost all test images

Lower the vigilance threshold:

```matlab
results = Chex_tester('chex_donut75', 'chex_train', 0.3);
```

### Stage 1 accepts all abnormal images

Raise the vigilance threshold or verify that `chex_train` contains only normal images. If the training distribution itself contains artefacts, the manifold will include them as normal.

### High memory usage

`MLP_chex` flattens each 390×320 image to 124 800 features. Training on the full 41 527-image set is memory-intensive. Reduce `batchSize` inside `MLP_chex.m` or work on a subset of `chex_train` if memory is limited.
