# Chest X-Ray Anomaly Detection — CheXpert Pipeline

A three-stage out-of-distribution (OOD) detection pipeline for chest radiographs, implemented in MATLAB. The system is trained exclusively on normal X-rays and flags abnormal images without requiring annotated abnormal examples during training.

---

## Repository Layout

```text
COEN498_OOD_Project/
    Chex_tester.m
    MD1_chex.m
    MD3_chex.m
    CNN_chex.m
    MLP_chex.m
    chex_cleanUp.m
    Utilities/
        chex_view_examples.m
        insert_donut_artifact.m
        insert_square_artifact.m
        pad_chex.py
    chex_train/          (41 527 normal JPEGs, 390×320 px)
    chex_donut75/        (3 000 artificially corrupted JPEGs)
    chex_squares75/      (3 000 artificially corrupted JPEGs)
    chex_pacemaker/      (3 535 real abnormal JPEGs)
    trained_models/
        cnn_chex_cache.mat
        mlp_chex_cache.mat
        md_chex_cache.mat
        stage3_chex_CNN_chex_16_32_64_128_fc256_lhl.mat
        stage3_chex_CNN_chex_16_32_64_128_fc256_fusion.mat
        stage3_chex_CNN_chex_16_32_64_128_fc256_mbm.mat
        stage3_chex_MLP_chex_1024_512_256_128_lhl.mat
        stage3_chex_MLP_chex_1024_512_256_128_fusion.mat
        stage3_chex_MLP_chex_1024_512_256_128_mbm.mat
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
| --- | --- | --- |
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

## Pipeline Overview

The pipeline is designed around the assumption that only normal images are available at training time. Abnormality is detected as deviation from the learned normal distribution.

```text
Test image
    │
    ▼
Stage 1 — Pixel-Space Pre-Filter  (MD1_chex)
    │  Builds a single PCA manifold from all normal training images.
    │  Rejects samples whose pixel-space MD confidence < vigilance.
    │  Active by default (vigilance = 0.5).
    │
    ▼  accepted samples only
Stage 2 — Regression Scoring  (CNN_chex + MLP_chex)
    │  Both networks trained with target label 1.0 for all normals.
    │  Score ≈ 1.0 → normal,  Score ≪ 1.0 → abnormal.
    │  Scores reported; no hard threshold applied at this stage.
    │
    ▼  accepted samples only
Stage 3 — Latent-Space Post-Filter  (MD3_chex × 2 networks × 3 algorithms)
    │  Mahalanobis distance in hidden-layer feature space.
    │  Three independent algorithms per network:
    │    Algorithm 1  LHL    — last hidden layer (relu5/CNN, relu4/MLP)
    │    Algorithm 2  FUSION — all layers z-normalised and concatenated
    │    Algorithm 3  MBM    — independent detector per architectural branch
    ▼
    6 Stage-3 accept/reject decisions  (CNN×3 + MLP×3)
```

Unlike the digit pipeline, Stage 1 is **active by default** for CheXpert. Pixel-space MD is effective here because clearly non-radiographic images differ grossly from chest X-rays at the pixel level.

---

## Quick Start

Set the MATLAB current folder to the repository root.

### Full three-stage pipeline

```matlab
results = Chex_tester('chex_train', 'chex_donut75');
results = Chex_tester('chex_train', 'chex_squares75');
results = Chex_tester('chex_train', 'chex_pacemaker');
```

Prompts to confirm or replace folder paths on each run.

### Custom vigilance threshold

```matlab
results = Chex_tester('chex_train', 'chex_donut75', 0.7);
```

### Disable Stage 1 filter

```matlab
results = Chex_tester('chex_train', 'chex_donut75', 0.5, false);
```

### Stage 1 pre-filter only

```matlab
s1 = MD1_chex('chex_donut75', 'chex_train', 0.5);
```

### Train / evaluate regression networks independently

```matlab
cnn = CNN_chex('chex_train');
mlp = MLP_chex('chex_train');
```

### Visualise examples

```matlab
chex_view_examples('chex_pacemaker');
```

### Clear all CheXpert caches

```matlab
chex_cleanUp();
```

---

## Scripts

### `Chex_tester.m` — End-to-End Driver

Runs the full three-stage pipeline on a test folder and prints a summary report.

**Workflow:**

1. Prompt for training (`chexRoot`) and test (`testFolder`) folder paths.
2. Run Stage 1 (`MD1_chex`): reject images far from the pixel-space manifold.
3. Run Stage 2 (`CNN_chex`, `MLP_chex`): compute regression anomaly scores on accepted images.
4. Run Stage 3 (`MD3_chex`): apply LHL, FUSION, and MBM detectors on accepted images for both networks.
5. Print per-stage acceptance counts and a final summary table.

**Call forms:**

```matlab
results = Chex_tester();
results = Chex_tester(chexRoot, testFolder);
results = Chex_tester(chexRoot, testFolder, rejectThreshold);
results = Chex_tester(chexRoot, testFolder, rejectThreshold, stage1Active);
```

`rejectThreshold` is the Stage 1 vigilance in `[0, 1]`, default `0.5`. `stage1Active` is logical, default `true`.

---

### `MD1_chex.m` — Stage 1: Pixel-Space Manifold Pre-Filter

Builds a single PCA manifold from the flattened pixel vectors of all normal training images. A test image is rejected if its Mahalanobis confidence falls below the vigilance threshold.

**Call forms:**

```matlab
s1 = MD1_chex(testFolder, chexRoot, vigilance);
s1 = MD1_chex(testFolder, chexRoot, vigilance, true);   % force retrain
```

**Cache:** `trained_models/md_chex_cache.mat`

---

### `CNN_chex.m` — Convolutional Regression Network

Trained as a one-class regressor: all normal training images have target label `1.0`. A score close to `1.0` indicates a normal X-ray; a lower score indicates anomaly.

**Architecture:**

```text
Input [320, 390, 1]
  Conv2D(3×3, 16 filters) + BatchNorm + ReLU (relu1) + MaxPool(2×2)  → [160, 195, 16]
  Conv2D(3×3, 32 filters) + BatchNorm + ReLU (relu2) + MaxPool(2×2)  → [ 80,  97, 32]
  Conv2D(3×3, 64 filters) + BatchNorm + ReLU (relu3) + MaxPool(2×2)  → [ 40,  48, 64]
  Conv2D(3×3,128 filters) + BatchNorm + ReLU (relu4) + MaxPool(2×2)  → [ 20,  24,128]
  Flatten → 61 440
  FC(256) + ReLU (relu5)  ← Stage 3 LHL feature source
  FC(1) → anomaly score (regression)
```

Training: Adam, learning rate `0.0001`, 5 epochs, batch size 32. 80/20 train/validation split.

**Stage 3 layer configuration:**

| Algorithm | Layers used |
| --- | --- |
| LHL | `relu5` |
| FUSION | `relu1, relu2, relu3, relu4, relu5` |
| MBM | `{relu1}` \| `{relu2}` \| `{relu3}` \| `{relu4, relu5}` |

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

```text
Input [320, 390, 1] → flatten → 124 800
  FC(1024) + ReLU (relu1)
  FC(512)  + ReLU (relu2)
  FC(256)  + ReLU (relu3)
  FC(128)  + ReLU (relu4)  ← Stage 3 LHL feature source
  FC(1) → anomaly score (regression)
```

Training: Adam, learning rate `0.0001`, 5 epochs, batch size 64.

**Stage 3 layer configuration:**

| Algorithm | Layers used |
| --- | --- |
| LHL | `relu4` |
| FUSION | `relu1, relu2, relu3, relu4` |
| MBM | `{relu1}` \| `{relu2}` \| `{relu3}` \| `{relu4}` |

**Call forms:**

```matlab
mlp = MLP_chex('chex_train');
mlp = MLP_chex('chex_train', true);   % force retrain
```

**Cache:** `trained_models/mlp_chex_cache.mat`

---

### `MD3_chex.m` — Stage 3: Latent-Space OOD Post-Filter

Fits a Mahalanobis distance model in the hidden-layer feature space of a trained network and scores test samples against it. One-class training: a single global mean and covariance are estimated from all normal training images. The OOD threshold is the 97.5th percentile of training-set distances.

Three algorithms are supported:

| Algorithm | Feature source | Detector |
| --- | --- | --- |
| `LHL` | Last hidden layer only | Single global covariance |
| `FUSION` | All hidden layers, z-normalised per layer, concatenated | Single global covariance on combined vector |
| `MBM` | Each architectural branch independently | One detector per branch; OOD if any fires |

**Call forms:**

```matlab
% Training mode — fit model on in-distribution features
model = MD3_chex('train', net, networkID, chexRoot, algo, layerConfig);

% Test mode — score new samples
s3 = MD3_chex('test', net, testImds, model);
```

**Caches:** `trained_models/stage3_chex_<networkID>_<algo>.mat`

---

### Utilities

| Script | Purpose |
| --- | --- |
| `chex_view_examples.m` | Display 10 random X-ray images from a folder in a 2×5 grid |
| `insert_donut_artifact.m` | Add circular artefacts to normal X-rays to create synthetic abnormal images |
| `insert_square_artifact.m` | Add rectangular artefacts to normal X-rays |
| `pad_chex.py` | Python — pad raw CheXpert images to 390×320 with black borders |
| `chex_cleanUp.m` | Delete all CheXpert `.mat` caches under `trained_models/` |

---

## Vigilance Threshold

The vigilance parameter (range `[0, 1]`) controls the Stage 1 acceptance boundary.

| Value | Effect |
| --- | --- |
| `0.0` | Accept all images — no filtering |
| `0.5` | Moderate filtering — recommended default |
| `1.0` | Maximum strictness — rejects nearly all images |

---

## Output Structures

### `Chex_tester` output

| Field | Description |
| --- | --- |
| `ChexRoot` | Training folder path used |
| `TestFolder` | Test folder path used |
| `Stage1RejectThreshold` | Stage 1 vigilance value |
| `Stage1` | Full `MD1_chex` result struct |
| `CNN.Stage2Scores` | Regression scores from the CNN on Stage-1-accepted images |
| `CNN.Network` | Trained CNN network object |
| `MLP.Stage2Scores` | Regression scores from the MLP |
| `MLP.Network` | Trained MLP network object |
| `Stage3.CNN.LHL` / `FUSION` / `MBM` | Stage 3 results per algorithm (CNN) |
| `Stage3.MLP.LHL` / `FUSION` / `MBM` | Stage 3 results per algorithm (MLP) |
| `TestFiles` | Cell array of test image file paths |

### `MD1_chex` output

| Field | Description |
| --- | --- |
| `NumSamples` | Total test images |
| `TestFiles` | Cell array of file paths |
| `Vigilance` | Threshold used |
| `MDScores` | Per-sample Mahalanobis confidence (higher = more normal) |
| `NormalizedDist` | Per-sample normalised Mahalanobis distance |
| `Accepted` / `IsOOD` | Logical masks |
| `AcceptedCount` / `RejectedCount` | Scalar counts |

### `CNN_chex` / `MLP_chex` output

| Field | Description |
| --- | --- |
| `Network` | Trained MATLAB network object |
| `TrainScores` | Regression scores on the training set |
| `TestScores` | Regression scores on the validation split |
| `RMSE` | Root mean squared error on the validation split |

---

## Caching and Reproducibility

All trained models are saved under `trained_models/` and reloaded on subsequent calls when the training folder path matches the cached value.

To force retraining:

```matlab
CNN_chex('chex_train', true);
MLP_chex('chex_train', true);
MD1_chex('chex_donut75', 'chex_train', 0.5, true);
```

To remove all CheXpert caches at once:

```matlab
chex_cleanUp();
```

---

## Troubleshooting

### Images fail to load

Verify that all JPEGs in the dataset folders are 390×320 grayscale. Run `pad_chex.py` if images have a different resolution.

### `trainNetwork` or `predict` not found

Install or enable the MATLAB Deep Learning Toolbox.

### Stage 1 rejects all or almost all test images

Lower the vigilance threshold:

```matlab
results = Chex_tester('chex_train', 'chex_donut75', 0.3);
```

### Stage 1 accepts all abnormal images

Raise the vigilance threshold or verify that `chex_train` contains only normal images.

### High memory usage with MLP

`MLP_chex` flattens each 390×320 image to 124 800 features. Training on the full 41 527-image set is memory-intensive. Reduce the batch size inside `MLP_chex.m` or train on a subset of `chex_train`.

### Stage 3 cache mismatch

If either regression network was retrained, delete the corresponding `stage3_chex_*.mat` files from `trained_models/` or run `chex_cleanUp()`.
