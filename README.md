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

Run the evaluation script to reproduce metrics on the held-out test split:

```bash
python scripts/evaluate.py --split test --figure reports/figures/confusion_matrix.png
```

The report is written to `reports/test_metrics.json` (accuracy, precision,
recall, F1, ROC-AUC, confusion matrix).

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
from Kaggle. Place the extracted `train`, `val`, and `test` folders under
`data/raw/`.

`scripts/make_splits.py` hashes every image, removes duplicates, drops any
train/val image that also appears in the test set, merges the tiny original
validation folder into train, and performs a seeded stratified re-split:

| Split | Images | NORMAL | PNEUMONIA |
|-------|--------|--------|-----------|
| train | 4,425  | 1,146  | 3,279     |
| val   | 781    | 202    | 579       |
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
