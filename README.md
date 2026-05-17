# 🔬 MIL Frameworks for Whole Slide Image Analysis

> A unified, modular deep learning framework for **Multiple Instance Learning (MIL)**-based Whole Slide Image (WSI) classification — integrating diverse MIL architectures, dataset split strategies, and comprehensive analysis tools.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![WSI](https://img.shields.io/badge/Task-WSI%20Classification-purple?style=flat-square)]()
[![MIL](https://img.shields.io/badge/Paradigm-Multiple%20Instance%20Learning-red?style=flat-square)]()

---

## Table of Contents

- [Background](#background)
- [Supported Tasks](#supported-tasks)
- [Implemented Metrics](#implemented-metrics)
- [Supported Dataset Split Methods](#supported-dataset-split-methods)
- [Pipeline Overview](#pipeline-overview)
  - [1. Data Collection](#1-data-collection)
  - [2. Mask Preparation](#2-mask-preparation)
  - [3. Feature Extraction](#3-feature-extraction)
  - [4. Dataset Preparation](#4-dataset-preparation)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Training](#training)
  - [Single CSV Training](#single-csv-training)
  - [K-Fold Split Training](#k-fold-split-training)
- [Evaluation & Inference](#evaluation--inference)
  - [Test with Labels](#test-with-labels)
  - [Inference without Labels](#inference-without-labels)
- [Visualisation](#visualisation)
  - [Attention Heatmaps](#attention-heatmaps)
  - [Feature Distribution Map](#feature-distribution-map)
- [Special Environments](#special-environments)
  - [PTC-MIL and LONG-MIL](#ptc-mil-and-long-mil)
  - [Mamba-2D-MIL](#mamba-2d-mil)
  - [Mamba-MIL](#mamba-mil)
- [Adding a New Model](#adding-a-new-model)
- [Passing Parameters via Shell Variables](#passing-parameters-via-shell-variables)
- [Acknowledgements](#acknowledgements)

---

## Background

**Whole Slide Images (WSIs)** are gigapixel-scale digital scans of tissue sections produced by whole slide scanners. A single WSI can contain billions of pixels, making direct deep learning classification impractical. WSI analysis typically involves tiling the slide into smaller patches and aggregating patch-level information to produce a slide-level prediction.

**Multiple Instance Learning (MIL)** is the dominant paradigm for WSI analysis. In MIL, each WSI is treated as a *bag* of patch-level *instances* (tiles). The bag-level label (e.g., tumour vs. normal) is known, but individual patch labels are not. MIL models learn to aggregate instance features into a slide-level representation for classification.

<p align="center">
  <img src="images/wsi_analysis.png" width="800" alt="MIL Framework Overview"/>
  <br/>
  <em>Figure 1: Overview of the MIL-based WSI classification pipeline.</em>
</p>

---

## Supported Tasks

| Task | Description |
|---|---|
| 🏷️ **WSI-level Classification** | Binary and multi-class slide-level prediction |
| 📊 **Bootstrapping AUROC Analysis** | Stratified bootstrap CI for AUC comparison |
| 📈 **Statistical Analysis** | Paired t-test, Wilcoxon test, balanced resampled t-test |
| 🎯 **Attention Analysis** | Attention score extraction and heatmap generation |
| 🗺️ **Feature Distribution Visualisation** | t-SNE plots of WSI-level feature embeddings |

---

## Implemented Metrics

| Metric | Variants |
|---|---|
| **AUC** | Macro · Micro · Weighted (equivalent for binary) |
| **F1 / Precision / Recall** | Macro · Micro · Weighted |
| **Accuracy** | Standard (`ACC`) · Balanced (`BACC` = Macro-Recall) |
| **Kappa** | Linear · Quadratic |
| **Confusion Matrix** | Full pixel-level matrix |

---

## Supported Dataset Split Methods

| Split Strategy | Description |
|---|---|
| User-defined Train–Val–Test | Fixed single split from one CSV |
| User-defined Train–Val | Fixed split, no test set |
| User-defined Train–Test | Fixed split, no validation set |
| Train–Val with K-Fold | K-fold cross-validation on train/val |
| Train–Val–Test with K-Fold | K-fold with held-out test set |
| Train–Val K-Fold then Test | K-fold training, final test on a fixed set |

> 📄 Full details on split differences: [`split_scripts/README.md`](split_scripts/README.md)

---

## Pipeline Overview

### 1. Data Collection

Collect WSIs for each class (e.g., Tumour and Normal). Organise them by class label for downstream processing.

### 2. Mask Preparation

Generate tissue masks for each WSI to guide tile extraction — ensuring only tissue regions are sampled and background is excluded.

### 3. Feature Extraction

Extract patch-level features using a pretrained encoder and save them as stacked `.pt` files (one per WSI). Store tile coordinates alongside features.

**Supported encoders include:** `CONCH` · `PLIP` · `ResNet50` · `ResNet18` · and others.

Each `.pt` file has shape `(N, feature_dim)` where `N` is the number of tiles per WSI.

**Required directory structure:**

```
/data/
├── WSIs/
│   ├── TCGA-A1-A0SB.svs
│   └── TCGA-A1-A0SD.svs
├── features/
│   ├── TCGA-A1-A0SB.pt          # Shape: (N, feature_dim)
│   └── TCGA-A1-A0SD.pt
└── coordinates/
    ├── TCGA-A1-A0SB_coordinates.npy   # Shape: (N, 2) → [[x, y], ...]
    └── TCGA-A1-A0SD_coordinates.npy
```

### 4. Dataset Preparation

Prepare a `Dataset.csv` with two columns:

| Column | Description |
|---|---|
| `slide_path` | Absolute path to the `.pt` feature file |
| `label` | Integer class label (`0` = Normal, `1` = Tumour) |

**Example:**

```
slide_path                                                                      label
/data/.../pt_files/P-0001.pt                                                    0
/data/.../pt_files/SHS-15-39995@E30-1.pt                                       1
```

Save `Dataset.csv` in `WSI_Classification/datasets/`.

Use the following notebooks in `split_scripts/` to prepare the CSV:

```
split_scripts/
├── dataset_preparation_mufasa.ipynb
└── dataset_preparation_others.ipynb
```

---

## Project Structure

```
WSI_Classification/
│
├── configs/                    ← YAML configuration files per MIL model
│   ├── MEAN_MIL.yaml
│   ├── CLAM_MB_MIL.yaml
│   └── ...
│
├── datasets/                   ← Dataset CSV files and split files
│   ├── Dataset.csv
│   ├── Total_5-fold_Stanford_1fold.csv
│   └── ...
│
├── logs/                       ← Training outputs (models, metrics, logs)
│   └── <DATASET>/<MODEL>/
│
├── modules/                    ← MIL model architecture definitions
│   ├── MEAN_MIL/
│   ├── CLAM_MB_MIL/
│   └── ...
│
├── process/                    ← Training process definitions per model
│   ├── process_all.py
│   ├── MEAN_MIL/
│   └── ...
│
├── split_scripts/              ← Dataset split utilities and notebooks
│   ├── README.md
│   ├── split_datasets_k_fold_train_val.py
│   └── ...
│
├── utils/                      ← Shared utilities (training loop, YAML reader, etc.)
│   ├── loop_utils.py
│   └── ...
│
├── vis_scripts/                ← Visualisation scripts (heatmaps, t-SNE)
│   ├── visualize_heatmaps_single.py
│   ├── visualize_heatmaps_batch.py
│   ├── draw_feature_map.py
│   └── validate_input_files.py
│
├── train_mil.py                ← Training entry point
├── test_inference_mil.py       ← Test / inference entry point
├── test_mil_calibration.py     ← Threshold calibration on test set
├── environment.yml             ← Conda environment specification
└── README.md
```

---

## Environment Setup

```bash
mkdir WSI_Classification
cd WSI_Classification

conda env create -f environment.yml
conda activate wsi_mil
```

> ⚠️ Some models (Mamba-MIL, LONG-MIL, PTC-MIL) require **separate conda environments** — see [Special Environments](#special-environments).

---

## Training

### YAML Configuration

Each model is configured via a YAML file in `configs/`. Key fields:

```yaml
General:
    MODEL_NAME: MEAN_MIL
    seed: 42
    num_classes: 2
    num_epochs: 200
    device: 0                        # GPU index

Dataset:
    DATASET_NAME: Stanford
    # Option A — single CSV file:
    dataset_csv_path: datasets/Stanford_Dataset.csv
    # Option B — k-fold split folder:
    # dataset_root_dir: datasets/Stanford/

    balanced_sampler:
        use: False
        replacement: True

Logs:
    log_root_dir: logs/

Model:
    in_dim: 1024                     # Feature dimension (encoder-dependent)
```

All YAML values can be overridden at the command line via `--options`.

---

### Single CSV Training

Train on all samples from a single CSV file (train = val = test — useful for sanity checks):

```bash
python train_mil.py \
    --yaml_path configs/MEAN_MIL.yaml \
    --options General.seed=42 \
              Dataset.DATASET_NAME=NSCLC \
              Dataset.feature_extractor=resnet50_1024 \
              Model.in_dim=1024 \
              Dataset.dataset_csv_path=/path/to/Total_5-fold_TCGA_NSCLC_1fold_with_path.csv \
              Logs.log_root_dir=/path/Output \
              General.num_epochs=200 \
              General.device=0
```

---

### K-Fold Split Training

#### Step 1 — Generate split files

```bash
cd WSI_Classification/split_scripts

python split_datasets_k_fold_train_val.py \
    --seed 2024 \
    --csv_path datasets/Dataset.csv \
    --save_dir datasets/ \
    --dataset_name Stanford \
    --k 5
```

This generates five stratified CSV files in `datasets/`:

```
datasets/
├── Total_5-fold_Stanford_1fold.csv
├── Total_5-fold_Stanford_2fold.csv
├── Total_5-fold_Stanford_3fold.csv
├── Total_5-fold_Stanford_4fold.csv
└── Total_5-fold_Stanford_5fold.csv
```

#### Step 2 — Update the YAML to use the split folder

```yaml
Dataset:
    DATASET_NAME: Stanford
    # dataset_csv_path: datasets/Dataset.csv    ← comment out
    dataset_root_dir: datasets/Stanford/        ← point to split folder
```

#### Step 3 — Train across all folds

```bash
python train_mil.py \
    --yaml_path configs/MEAN_MIL.yaml \
    --options Dataset.DATASET_NAME=TCGA_NSCLC \
              Dataset.feature_extractor=resnet50_1024 \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir=datasets/TCGA_NSCLC/TCGA_NSCLC_HISTOLAB_resnet50_1024_splits \
              Logs.log_root_dir=logs/ \
              General.seed=42 \
              General.num_epochs=200 \
              General.device=0
```

**Training outputs** are saved to `logs/<DATASET>/<MODEL>/`:

```
logs/TCGA_NSCLC/MEAN_MIL/
└── <timestamp>/
    ├── fold_1/
    │   ├── Best_EPOCH_XXX.pth
    │   ├── metrics.csv
    │   └── model_fold1.log
    ├── fold_2/ … fold_5/
    └── summary.csv
```

---

## Evaluation & Inference

### Test with Labels

Evaluate a trained model on a labelled test or validation set:

```bash
python test_inference_mil.py \
    --yaml_path configs/MEAN_MIL.yaml \
    --model_weight_path /path/to/Best_EPOCH_XXX.pth \
    --options General.seed=42 \
              Dataset.DATASET_NAME=NSCLC \
              Dataset.group=val \
              Dataset.dataset_csv_path=/path/to/split.csv \
              Logs.log_root_dir=/path/Output
```

| Argument | Description |
|---|---|
| `--yaml_path` | MIL model configuration file |
| `--model_weight_path` | Path to trained `.pth` checkpoint |
| `Dataset.group` | `val` or `test` |
| `Dataset.dataset_csv_path` | CSV with columns `val_slide_path` + `val_label` (or `test_slide_path` + `test_label`) |
| `Logs.log_root_dir` | Output directory |

**Output:**

```
Output/
├── summary.txt                           ← Full console log
├── config_MEAN_MIL.yaml                  ← Applied configuration snapshot
├── <csv_filename>.csv                    ← Copy of input dataset CSV
└── Detailed_Probabilities_MEAN_MIL.csv  ← Predicted probabilities, GT labels, predicted classes
```

---

### Inference without Labels

Run inference on unlabelled slides (no ground truth required):

```bash
python test_inference_mil.py \
    --yaml_path configs/MEAN_MIL.yaml \
    --model_weight_path /path/to/Best_EPOCH_XXX.pth \
    --options General.seed=42 \
              Dataset.DATASET_NAME=NSCLC \
              Dataset.group=test \
              Dataset.dataset_csv_path=/path/to/slides.csv \
              Logs.log_root_dir=/path/Output
```

> The CSV should contain only one column: `test_slide_path` — one `.pt` file path per row.

**Output:** Same structure as above; ground truth column will be absent.

---

## Visualisation

### Attention Heatmaps

Generate slide-level attention heatmaps for models that use attention mechanisms.

#### Validate input files

```bash
python vis_scripts/validate_input_files.py \
    --feature_path /data/features/WSI_NAME.pt \
    --coords_path /data/coordinates/WSI_NAME_coordinates.npy
```

#### Single WSI

```bash
python vis_scripts/visualize_heatmaps_single.py \
    --config single_vis_config.yaml \
    --wsi_dir /data/WSIs/Tumor \
    --feature_dir /data/features/resnet50_1024 \
    --coords_dir /data/coordinates \
    --wsi_name SLIDE_NAME \
    --model_name AC_MIL \
    --model_config configs/AC_MIL.yaml \
    --checkpoint /path/to/Best_EPOCH_26.pth \
    --output_dir output
```

**Output structure:**

```
output/AC_MIL/
├── WSI_NAME1/
│   ├── config_WSI_NAME1.yaml
│   ├── WSI_NAME1_attention.npy      ← Raw attention scores
│   ├── WSI_NAME1_attention.txt
│   ├── WSI_NAME1_original.png       ← Original WSI thumbnail
│   ├── WSI_NAME1_heatmap_only.png   ← Attention heatmap only
│   ├── WSI_NAME1_heatmap.png        ← Overlay: WSI + heatmap
│   └── WSI_NAME1_summary.png        ← Summary figure
├── WSI_NAME2/
└── WSI_NAME3/
```

#### Batch Processing

```bash
python vis_scripts/visualize_heatmaps_batch.py \
    --config batch_vis_config.yaml \
    --wsi_dir /data/WSIs/Tumor \
    --feature_dir /data/features/resnet50_1024 \
    --coords_dir /data/coordinates \
    --model_name AC_MIL \
    --model_config configs/AC_MIL.yaml \
    --checkpoint /path/to/Best_EPOCH_26.pth \
    --output_dir output
```

---

### Feature Distribution Map

Visualise the t-SNE distribution of test-set WSI feature embeddings, coloured by class:

```bash
python vis_scripts/draw_feature_map.py \
    --yaml_path configs/CLAM_MB_MIL.yaml \
    --ckpt_path /path/to/Best_EPOCH_176.pth \
    --id2class '{0:"Normal", 1:"Tumor"}' \
    --save_path /path/to/output_dir \
    --test_dataset_csv /path/to/split.csv \
    --data_split val \
    --seed 2023
```

---

## Special Environments

Some models require dependencies that conflict with the main environment. Create dedicated conda environments as described below.

### PTC-MIL and LONG-MIL

```bash
conda create -n transformer python=3.10 -y
conda activate transformer

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.20
pip install addict ruamel.yaml pytz pandas h5py PyYAML einops tqdm easydict
pip install "numpy<2.0"
pip install mmcv-full==1.7.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

```bash
python train_mil.py \
    --yaml_path configs/LONG_MIL.yaml \
    --options Dataset.DATASET_NAME=CAMELYON16 \
              Dataset.feature_extractor=resnet50_1024 \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir=datasets/CAMELYON16/splits \
              Logs.log_root_dir=logs/Test \
              General.seed=2023 General.num_epochs=1 General.device=2
```

---

### Mamba-2D-MIL

```bash
conda create -n mambamil_2d python=3.10 -y
conda activate mambamil_2d

pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install packaging ninja
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.2 --no-build-isolation
pip install transformers==4.36.2 --no-cache-dir
pip install addict ruamel.yaml pytz pandas h5py PyYAML
```

```bash
python train_mil.py \
    --yaml_path configs/MAMBA2D_MIL.yaml \
    --options Dataset.DATASET_NAME=CAMELYON16 \
              Dataset.feature_extractor=resnet50_1024 \
              Model.in_dim=1024 \
              Dataset.dataset_root_dir=datasets/CAMELYON16/splits \
              Logs.log_root_dir=logs/Test \
              General.seed=2023 General.num_epochs=1 General.device=7
```

---

### Mamba-MIL

```bash
conda create -n mambamil python=3.10 -y
conda activate mambamil

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install packaging setuptools
pip install causal-conv1d==1.1.1 --no-build-isolation --no-cache-dir

# Install mamba from source
cd modules/MAMBA_MIL/mamba
python -m pip install --no-build-isolation .
cd ../../..

pip install scikit-survival==0.22.2 pandas==2.2.1 lifelines \
    addict ruamel.yaml h5py PyYAML tqdm einops
```

**Supported model variants:**

```bash
# MAMBA_MIL
python train_mil.py --yaml_path configs/MAMBA_MIL.yaml \
    --options General.MODEL_NAME=MAMBA_MIL General.num_epochs=2 General.device=7

# BiMAMBA_MIL
python train_mil.py --yaml_path configs/MAMBA_MIL.yaml \
    --options General.MODEL_NAME=BiMAMBA_MIL General.num_epochs=1 General.device=7

# SRMAMBA_MIL
python train_mil.py --yaml_path configs/MAMBA_MIL.yaml \
    --options General.MODEL_NAME=SRMAMBA_MIL General.num_epochs=1 General.device=7
```

---

## Adding a New Model

Only **4 steps** — `train_mil.py` and `test_inference_mil.py` require **zero modifications**.

**Step 1** — Register in `process/process_all.py`:

```python
if args.General.MODEL_NAME == 'DeepAttn_MIL':
    from .DeepAttn_MIL.process_DeepAttn_mil import process_DeepAttn_MIL
    process_DeepAttn_MIL(args)
```

**Step 2** — Create process files:

```
process/DeepAttn_MIL/
├── process_DeepAttn_mil.py
└── __init__.py
```

> If your model needs a custom training loop, add it to `utils/loop_utils.py`.

**Step 3** — Create model architecture:

```
modules/DeepAttn_MIL/
└── deep_attn_mil.py
```

**Step 4** — Create a YAML config:

```
configs/DeepAttn_MIL.yaml     ← copy from unet.yaml, set model.name: DeepAttn_MIL
```

Then train:

```bash
python train_mil.py --yaml_path configs/DeepAttn_MIL.yaml
```

---

## Passing Parameters via Shell Variables

Use shell variables for cleaner multi-run scripts:

```bash
pre_processing=CLAM
dataset=TCGA_LUAD
model=CLAM_MB_MIL

python train_mil.py \
    --yaml_path configs/${model}.yaml \
    --options Dataset.DATASET_NAME=${dataset} \
              Dataset.feature_extractor=resnet18 \
              Model.in_dim=512 \
              Dataset.dataset_root_dir=datasets/${dataset}/${dataset}_${pre_processing}_resnet18 \
              Logs.log_root_dir=logs/${pre_processing} \
              General.num_epochs=20
```

---

## Acknowledgements

This framework builds upon and is inspired by the following open-source repositories:

- [CLAM](https://github.com/mahmoodlab/CLAM) — Clustering-constrained Attention MIL
- [MIL_BASELINE](https://github.com/rathinaraja/MIL_BASELINE) — Unified MIL baseline implementations
