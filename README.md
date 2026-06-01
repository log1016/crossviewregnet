# CrossViewRegNet

Official implementation for **CrossViewRegNet: a multi-view deep learning method for tree species classification using only point cloud coordinate information**.

CrossViewRegNet converts each individual-tree point cloud into seven single-channel projection images: one top view, four side views, one bottom view, and one DBH slice view. A shared DenseNet-201 backbone extracts view-wise features, tree height is encoded by an MLP branch, and a cross-view attention fusion block aggregates all features for species classification.

## Repository Structure

```text
CrossViewRegNet_public_20260515/
  crossviewregnet/
    dataset.py        # NPZ dataset loader
    io.py             # LAS/LAZ reader
    model.py          # CrossViewRegNet architecture
    projection.py     # point cloud to multi-view image projection
    utils.py          # transforms, metrics, reproducibility helpers
  scripts/
    preprocess.py     # LAS/LAZ to NPZ conversion
    train.py          # model training and validation
    predict.py        # checkpoint inference and CSV export
  requirements.txt
  .gitignore
```

## Installation

```bash
conda create -n crossviewregnet python=3.10 -y
conda activate crossviewregnet
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version if the default `pip` build is not suitable for your machine.

## Expected Data Format

The metadata CSV files should contain at least:

```text
filename,species_id,tree_H
/train/00070.npz,8,9.018
```

For training, `species_id` is required. For prediction-only CSV files, `species_id` can be omitted. The `filename` column is used by basename, so `/train/00070.npz`, `00070.npz`, and `00070.laz` all map to `00070.npz` after preprocessing.

The class lookup CSV should contain:

```text
species_id,species
8,Example_species
```

## Preprocess Point Clouds

```bash
python scripts/preprocess.py \
  --input-dir /path/to/raw_las_or_laz \
  --metadata-csv /path/to/train_labels.csv \
  --output-dir /path/to/npz/train \
  --resolution 256 \
  --workers 16
```

Each output NPZ contains:

```text
images: float32 array with shape [7, H, W]
height: scalar tree height
```

## Train

```bash
python scripts/train.py \
  --train-csv /path/to/train_labels.csv \
  --val-csv /path/to/vali_labels.csv \
  --data-dir /path/to/npz/train \
  --output-dir /path/to/outputs \
  --num-classes 33 \
  --batch-size 16 \
  --samples-per-epoch 8192 \
  --epochs 1000 \
  --patience 15
```

Training writes the best checkpoint, per-epoch metrics, a loss curve, and a JSON summary to `--output-dir`.

## Predict

Regular prediction output:

```bash
python scripts/predict.py \
  --test-csv /path/to/test_labels.csv \
  --data-dir /path/to/npz/test \
  --checkpoint /path/to/best_model.pth \
  --lookup-csv /path/to/lookup.csv \
  --train-csv /path/to/train_labels.csv \
  --output-csv predictions.csv \
  --tta 1
```

Public benchmark submission format:

```bash
python scripts/predict.py \
  --test-csv /path/to/test_labels.csv \
  --data-dir /path/to/npz/test \
  --checkpoint /path/to/best_model.pth \
  --lookup-csv /path/to/lookup.csv \
  --train-csv /path/to/train_labels.csv \
  --output-csv submission.csv \
  --submission-format \
  --tta 50
```

With `--submission-format`, the output columns are:

```text
treeID,predicted_species
69,Example_species
```

`treeID` is extracted from the numeric part of the filename and saved as an integer, so leading zeros are removed.

## Notes

- Only XYZ coordinates are used from LAS/LAZ files.
- DBH slice extraction uses points between 1.0 m and 1.5 m by default. If too few points are available in this range, the DBH view is stored as a zero image.
- Test-time augmentation is optional. `--tta 1` is recommended for fast deployment; larger values can improve stability at higher inference cost.

## Data and Checkpoints

The public coordinate-normalized Ts3D dataset is available from the GitHub release page:

- [Ts3D public coordinate-normalized data](https://github.com/log1016/crossviewregnet/releases/tag/ts3d-public-v1)

The released LAS files remove absolute geographic/projected coordinates by centering each individual tree in local coordinates while preserving its 3D structure and metric height.
