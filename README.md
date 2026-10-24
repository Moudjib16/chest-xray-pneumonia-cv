# chest-xray-pneumonia-cv

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv&logoColor=white)

End-to-end chest X-ray classification (NORMAL vs PNEUMONIA) with a fine-tuned
ResNet18, Grad-CAM explanations, and a Streamlit demo.

## Overview

* Transfer learning from ImageNet-pretrained ResNet18 to a 2-class classifier.
* Albumentations augmentation pipeline (resize, horizontal flip, rotation,
  ImageNet normalisation).
* Class-weighted cross-entropy to counter the pneumonia-heavy dataset.
* Grad-CAM heatmaps to show which regions drive each prediction.
* Streamlit app for upload-and-explain inference.

## Results

Metrics at a 0.5 decision threshold
(`python scripts/evaluate.py --split val|test`):

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| val   | 0.9823   | 0.9979    | 0.9774 | 0.9876 | 0.9989 |
| test  | 0.9223   | 0.8970    | 0.9897 | 0.9410 | 0.9684 |

Validation is now an honest estimate: the split is patient- and
near-duplicate-aware, so it cannot share a patient with training. The test
column is the official Kaggle test folder, which is a harder, shifted external
set — the model keeps high sensitivity (98.97%, only 4 missed pneumonias) with
89.70% precision there. Earlier checkpoints trained on a leaky split scored
99.4% on validation but only 73.5% on test; fixing the split and adding
stronger augmentation closed most of that gap.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
from Kaggle. Place the extracted `train`, `val`, and `test` folders under
`data/raw/`.

`scripts/make_splits.py` merges the tiny original validation folder into the
training pool, then builds a **leak-free** split:

* exact duplicates are removed by MD5;
* images are grouped by patient/study identifier parsed from the filename;
* near-duplicates are grouped with a 255-bit difference hash, so resized or
  recompressed copies never cross splits;
* any pooled image sharing a patient with the official test folder is dropped.

The result is a seeded, class-stratified group split with zero patient overlap
between splits:

| Split | Images | NORMAL | PNEUMONIA |
|-------|--------|--------|-----------|
| train | 4,074  | 1,156  | 2,918     |
| val   | 679    | 192    | 487       |
| test  | 618    | 231    | 387       |

## Project structure

```
app/main.py             Streamlit upload-and-explain demo
configs/config.yaml     paths, hyperparameters, CPU/GPU profiles
notebooks/01_eda.ipynb  exploratory data analysis
scripts/make_splits.py  de-duplicate images and build data/splits.csv
scripts/sanity_check.py validate paths, overlaps, and batch loading
scripts/evaluate.py     test metrics and confusion matrix
src/dataset.py          Dataset, transforms, DataLoaders
src/model.py            ResNet18 builder and Grad-CAM target layer
src/train.py            CPU/GPU training loop with best-checkpoint saving
src/gradcam.py          Grad-CAM heatmap generation and overlay helpers
tests/                  dataset, model, and Grad-CAM tests
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Training uses a ResNet18 initialised with ImageNet weights, so the first run
needs to download them (cached afterwards in the PyTorch hub cache).

## Usage

```bash
# build de-duplicated, stratified splits
python scripts/make_splits.py

# verify the pipeline
python scripts/sanity_check.py

# train (CPU profile by default; also --profile gpu, --epochs N)
python src/train.py --profile cpu

# evaluate the best checkpoint
python scripts/evaluate.py --split test

# tests
pytest -q

# run the demo
streamlit run app/main.py
```

Hyperparameters and split paths live in `configs/config.yaml`. The best
checkpoint is saved to `models/resnet18_xray.pth`.

## Grad-CAM

`src/gradcam.py` hooks the last convolutional block (`model.layer4`), averages
the gradients of the target class over the feature map, weights the activations,
applies ReLU, and upsamples the result to the input resolution. The Streamlit
app overlays the heatmap (JET colormap) on the uploaded X-ray.

## Disclaimer

This project is for research and portfolio demonstration only. It is not a
medical device and must not be used for clinical decisions.
