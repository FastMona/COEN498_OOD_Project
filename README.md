# COEN498_OOD_Project

Out-of-distribution (OOD) filtering and classification pipeline built in MATLAB for handwritten image datasets in IDX format.

The project combines three components:

1. `MD_filter` for manifold-based OOD rejection.
2. `CNN_reader` for CNN classification.
3. `MLP_reader` for fully connected MLP classification.

`Folder_testor` ties all three together by rejecting OOD samples first, then classifying only accepted samples with CNN and MLP.

## Repository Layout

Expected top-level structure:

```text
COEN498_OOD_Project/
    CNN_reader.m
    MLP_reader.m
    MD_filter.m
    Folder_testor.m
    visualize_examples.m
    visualize_ood_examples.m
    MNIST_digits/raw/
        train-images-idx3-ubyte
        train-labels-idx1-ubyte
        t10k-images-idx3-ubyte
        t10k-labels-idx1-ubyte
    KMNIST_japanese/
        t10k-images-idx3-ubyte
        t10k-labels-idx1-ubyte
    trained_models/
        cnn_reader_cache.mat
        mlp_reader_cache.mat
        md_filter_cache.mat
        folder_paths_cache.mat
```

The provided dataset folders in this repository already follow the naming expected by the scripts.

## Requirements

- MATLAB (R2021b+ recommended)
- Deep Learning Toolbox (required for `trainNetwork`, `classify`, `confusionchart`)
- Image Processing Toolbox is recommended for `imshow` in visualization scripts

No Python environment is required for the MATLAB pipeline itself.

## Quick Start

From MATLAB, set the current folder to the repository root.

### 1) Run the full OOD + classification pipeline

```matlab
results = Folder_testor();
```

Default behavior:

- OOD threshold (`rejectThreshold` / vigilance): `0.5`
- Training/test folder roots are resolved by `getSetFolderPaths.m`
  - uses saved paths from `trained_models/folder_paths_cache.mat` when available
  - otherwise falls back to code defaults (`MNIST_digits/raw` and `KMNIST_japanese`) and stores them
  - no runtime path prompt is required

### 2) Run only the manifold OOD filter

```matlab
md = MD_filter('KMNIST_japanese', 0.5, false, true);
```

### 3) Train/evaluate individual classifiers

```matlab
cnn = CNN_reader('MNIST_digits/raw');
mlp = MLP_reader('MNIST_digits/raw');
```

### 4) Visualization utilities

```matlab
visualize_examples('MNIST_fashion');
visualize_ood_examples('KMNIST_japanese', 0.5, 5);
```

## Script Overview

### `MD_filter.m`

Manifold-distance OOD detector trained on MNIST training data (`MNIST_digits/raw`).

- Accepts folder input (IDX test set) or numeric input (`28x28`, `Nx784`, `28x28xN`, `28x28x1xN`)
- Produces per-sample:
  - predicted nearest digit manifold (`BestDigit`)
  - confidence score (`Confidence`)
  - accepted/rejected mask (`Accepted`, `IsOOD`)
- Uses vigilance threshold in `[0, 1]` where higher values are stricter
- Caches trained manifold model in `trained_models/md_filter_cache.mat`

Key call forms:

```matlab
MD_filter();                                   % train/load model only
res = MD_filter(testInput);                    % score with default vigilance=0.5
res = MD_filter(testInput, vigilance);         % custom threshold
res = MD_filter(testInput, vig, true, false);  % force retrain, silent mode
```

### `CNN_reader.m`

CNN classifier for IDX datasets (digit labels by default, fashion labels if path contains `fashion`).

- Architecture: 2 conv blocks + FC layers
- Trains on training IDX files and evaluates on test IDX files in the given root
- Returns network, predictions, confusion matrix/table, and accuracy
- Caches model in `trained_models/cnn_reader_cache.mat`

Key call forms:

```matlab
cnn = CNN_reader();
cnn = CNN_reader('MNIST_digits/raw');
cnn = CNN_reader('MNIST_digits/raw', true); % force retrain
```

### `MLP_reader.m`

Fully connected classifier operating on flattened `784`-dim vectors.

- Default hidden layers: `[512, 256, 128]`
- Hidden layer widths can be edited inside the file (`hiddenLayerSizes`)
- Returns the same output fields as `CNN_reader`
- Caches model in `trained_models/mlp_reader_cache.mat`

Key call forms:

```matlab
mlp = MLP_reader();
mlp = MLP_reader('MNIST_digits/raw');
mlp = MLP_reader('MNIST_digits/raw', true); % force retrain
```

### `Folder_testor.m`

End-to-end evaluator.

Workflow:

1. Apply `MD_filter` to candidate/OOD folder.
2. Keep only accepted samples.
3. Run CNN and MLP on accepted samples.
4. Report agreement and accepted-sample digit distribution.

Key call forms:

```matlab
results = Folder_testor();
results = Folder_testor(trainFolder, testFolder);
results = Folder_testor(trainFolder, testFolder, 0.5);
```

Notes:

- `trainFolder` and `testFolder` are optional.
- If omitted, saved/default paths from `getSetFolderPaths` are used.
- Console output always reports:
  - `loading training data from: ...`
  - `Running inference test on: ...`

### `visualize_examples.m`

Displays up to 10 random images from an IDX test set folder (`t10k-*` files), with labels when available.

### `visualize_ood_examples.m`

Runs `MD_filter` and shows accepted (top row) vs rejected (bottom row) examples.

## Output Structures

### `Folder_testor` output (high-level)

- `results.MDFilter`: full OOD filter result struct
- `results.CNN.YPred`: CNN predictions for accepted samples
- `results.MLP.YPred`: MLP predictions for accepted samples
- `results.AgreementAcceptedPct`: CNN/MLP agreement on accepted subset
- `results.DigitDistribution`: table of accepted-sample predicted counts

### `MD_filter` output (score mode)

- `BestDigit`, `BestDistance`, `Confidence`
- `Accepted`, `IsOOD`
- `AcceptedCount`, `RejectedCount`
- `DistanceMatrix`, `ConfidenceMatrix`
- `Images4D`, `Features`, `InputLabels`

## Caching and Reproducibility

Each major model is cached under `trained_models/`.

- Repeated runs reuse cache when training data root (and MLP architecture) match.
- Set `forceRetrain=true` in `CNN_reader`, `MLP_reader`, or `MD_filter` to retrain.
- Folder roots are cached in `trained_models/folder_paths_cache.mat`.

To clear all caches manually, delete:

- `trained_models/cnn_reader_cache.mat`
- `trained_models/mlp_reader_cache.mat`
- `trained_models/md_filter_cache.mat`
- `trained_models/folder_paths_cache.mat`

Or run:

```matlab
cleanUp();
```

## Troubleshooting

### Missing IDX file errors

Ensure each dataset folder contains matching IDX image and label files with exact names:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

### Toolbox-related errors (`trainNetwork`, `confusionchart`, etc.)

Install/enable MATLAB Deep Learning Toolbox.

### No accepted OOD samples in `Folder_testor`

Lower the threshold, for example:

```matlab
results = Folder_testor([], [], 0.4);
```

## Notes

- Default classifier training data is `MNIST_digits/raw`.
- `CNN_reader` and `MLP_reader` can also be pointed to Fashion-MNIST-style IDX roots.
- Visualization scripts are intended for quick qualitative checks, not benchmarking.
