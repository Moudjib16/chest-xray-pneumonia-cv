"""Sanity-check the chest X-ray data pipeline.

Prints class distributions, validates that every referenced image exists, checks
for duplicate/overlapping paths across splits, and renders one training batch to
``reports/figures/sample_batch.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import (  # noqa: E402
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_dataloaders,
    load_config,
)

FIGURES_DIR = ROOT / "reports" / "figures"


def print_distribution(frame: pd.DataFrame) -> None:
    counts = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["pneumonia_pct"] = (counts["PNEUMONIA"] / counts["total"] * 100).round(2)
    print("\nClass distribution per split:")
    print(counts.to_string())


def find_missing(frame: pd.DataFrame) -> list[str]:
    return [path for path in frame["path"] if not (ROOT / path).is_file()]


def find_overlaps(frame: pd.DataFrame) -> list[str]:
    problems: list[str] = []

    duplicates = frame.loc[frame["path"].duplicated(), "path"].unique()
    if len(duplicates) > 0:
        problems.append(f"{len(duplicates)} duplicate paths appear more than once")

    splits = frame["split"].unique().tolist()
    for i, first in enumerate(splits):
        for second in splits[i + 1 :]:
            first_paths = set(frame.loc[frame["split"] == first, "path"])
            second_paths = set(frame.loc[frame["split"] == second, "path"])
            shared = first_paths & second_paths
            if shared:
                problems.append(f"{len(shared)} paths shared between {first} and {second}")

    return problems


def denormalize(image) -> np.ndarray:
    array = image.permute(1, 2, 0).numpy()
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(array, 0.0, 1.0)


def save_batch_grid(loader, out_path: Path, columns: int = 4) -> None:
    images, labels = next(iter(loader))
    count = min(len(images), columns * columns)
    rows = int(np.ceil(count / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 2.5))
    for index, axis in enumerate(np.ravel(axes)):
        axis.axis("off")
        if index < count:
            axis.imshow(denormalize(images[index]))
            axis.set_title(CLASSES[int(labels[index])], fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved sample batch to {out_path.relative_to(ROOT)}")
    print(f"batch tensor: {tuple(images.shape)} {images.dtype}")


def main() -> int:
    config = load_config()
    frame = pd.read_csv(ROOT / config["common"]["splits_csv"], dtype={"path": str, "label": str})

    print(f"Loaded {len(frame)} rows from {config['common']['splits_csv']}")
    print("\nRows per split:")
    print(frame["split"].value_counts().to_string())
    print_distribution(frame)

    missing = find_missing(frame)
    overlaps = find_overlaps(frame)

    print("\nChecks:")
    print(f"  missing files: {len(missing)}")
    for path in missing[:5]:
        print(f"    - {path}")
    print(f"  duplicate/overlap issues: {len(overlaps)}")
    for problem in overlaps:
        print(f"    - {problem}")

    if missing or overlaps:
        return 1

    loader = build_dataloaders(split="train", config=config)
    save_batch_grid(loader, FIGURES_DIR / "sample_batch.png")
    print("\nPipeline OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
