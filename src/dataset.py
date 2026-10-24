"""PyTorch Dataset and DataLoader utilities for the chest X-ray dataset."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("NORMAL", "PNEUMONIA")
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_config(path: Path = ROOT / "configs" / "config.yaml") -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_transforms(img_size: int, train: bool) -> A.Compose:
    """Return the Albumentations pipeline for training or evaluation."""
    if train:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class ChestXrayDataset(Dataset):
    """Load chest X-ray images listed in ``data/splits.csv``.

    Each item is a ``(image_tensor, label)`` pair where the image is a
    normalised ``float32`` tensor of shape ``(3, H, W)`` and the label is
    ``0`` for NORMAL and ``1`` for PNEUMONIA. The file paths are also kept in
    ``self.paths`` for evaluation and visualisation tooling.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: A.Compose | None = None,
        root: Path = ROOT,
    ) -> None:
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].map(LABEL_TO_INDEX).astype(int).tolist()
        self.transform = transform
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = cv2.imread(str(self.root / self.paths[index]))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, self.labels[index]


def build_dataloaders(
    profile: str = "cpu",
    split: str = "train",
    config: dict | None = None,
    root: Path = ROOT,
) -> DataLoader:
    """Build a DataLoader for ``split`` using the given config profile."""
    config = config or load_config()
    common = config["common"]
    profile_config = config[profile]

    frame = pd.read_csv(root / common["splits_csv"], dtype={"path": str, "label": str})
    frame = frame[frame["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows found for split '{split}'")

    train = split == "train"
    dataset = ChestXrayDataset(
        frame,
        transform=build_transforms(common["img_size"], train=train),
        root=root,
    )

    num_workers = profile_config["num_workers"]
    return DataLoader(
        dataset,
        batch_size=profile_config["batch_size"],
        shuffle=train,
        num_workers=num_workers,
        pin_memory=profile_config["pin_memory"],
        persistent_workers=num_workers > 0,
    )
