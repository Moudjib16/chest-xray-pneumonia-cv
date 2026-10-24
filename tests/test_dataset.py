"""Tests for the chest X-ray dataset, transforms, and dataloaders."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import pandas as pd
import pytest
import torch

from src.dataset import (
    ChestXrayDataset,
    LABEL_TO_INDEX,
    build_dataloaders,
    build_transforms,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()
SPLITS_CSV = ROOT / CONFIG["common"]["splits_csv"]

pytestmark = pytest.mark.skipif(
    not SPLITS_CSV.exists(),
    reason="data/splits.csv is missing; run scripts/make_splits.py first",
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_csv(SPLITS_CSV, dtype={"path": str, "label": str})


@pytest.fixture(scope="module")
def raw_rgb_image(frame: pd.DataFrame):
    path = ROOT / frame.loc[frame["split"] == "train", "path"].iloc[0]
    image = cv2.imread(str(path))
    assert image is not None, f"could not read {path}"
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.fixture
def fast_config() -> dict:
    config = deepcopy(CONFIG)
    config["cpu"]["num_workers"] = 0
    config["cpu"]["batch_size"] = 4
    return config


def test_label_mapping() -> None:
    assert LABEL_TO_INDEX == {"NORMAL": 0, "PNEUMONIA": 1}


def test_eval_transform_shape_and_dtype(raw_rgb_image) -> None:
    output = build_transforms(224, train=False)(image=raw_rgb_image)["image"]
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32


def test_eval_transform_is_deterministic(raw_rgb_image) -> None:
    transform = build_transforms(224, train=False)
    first = transform(image=raw_rgb_image)["image"]
    second = transform(image=raw_rgb_image)["image"]
    assert torch.equal(first, second)


def test_train_transform_is_stochastic(raw_rgb_image) -> None:
    transform = build_transforms(224, train=True)
    outputs = [transform(image=raw_rgb_image)["image"] for _ in range(20)]
    assert any(not torch.equal(outputs[0], other) for other in outputs[1:])


def test_dataset_length(frame: pd.DataFrame) -> None:
    rows = frame[frame["split"] == "train"].head(10).reset_index(drop=True)
    dataset = ChestXrayDataset(rows, transform=build_transforms(224, train=False))
    assert len(dataset) == 10
    assert len(dataset.paths) == 10


def test_getitem_returns_tensor_and_label(frame: pd.DataFrame) -> None:
    rows = frame[frame["split"] == "train"].groupby("label").head(1).reset_index(drop=True)
    dataset = ChestXrayDataset(rows, transform=build_transforms(224, train=False))

    assert set(dataset.labels) == {0, 1}
    for index in range(len(dataset)):
        image, label = dataset[index]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32
        assert isinstance(label, int)
        assert label in {0, 1}


def test_dataloader_batch_shapes(fast_config: dict) -> None:
    loader = build_dataloaders(split="train", config=fast_config)
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 224, 224)
    assert images.dtype == torch.float32
    assert labels.shape == (4,)
    assert labels.dtype == torch.int64


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dataloader_split_sizes(fast_config: dict, frame: pd.DataFrame, split: str) -> None:
    loader = build_dataloaders(split=split, config=fast_config)
    assert len(loader.dataset) == int((frame["split"] == split).sum())


def test_unknown_split_raises(fast_config: dict) -> None:
    with pytest.raises(ValueError):
        build_dataloaders(split="does-not-exist", config=fast_config)
