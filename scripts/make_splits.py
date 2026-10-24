"""Build data/splits.csv from the raw chest X-ray dataset.

Merges the tiny Kaggle ``val`` split into ``train`` and performs a seeded,
stratified re-split so the validation set is large enough to be meaningful.
The ``test`` split is copied through untouched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}

CLASSES = ("NORMAL", "PNEUMONIA")


def load_config() -> dict:
    with (ROOT / "configs" / "config.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["common"]


def collect_images(split_dir: Path, split_names: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for split_name in split_names:
        for label in CLASSES:
            class_dir = split_dir / split_name / label
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "path": path.relative_to(ROOT).as_posix(),
                            "label": label,
                        }
                    )
    return rows


def main() -> None:
    config = load_config()
    data_dir = ROOT / config["data_dir"]
    splits_csv = ROOT / config["splits_csv"]
    seed = int(config["seed"])
    val_fraction = float(config["val_fraction"])

    pooled = collect_images(data_dir, ("train", "val"))
    if not pooled:
        raise SystemExit(f"No images found under {data_dir / 'train'} or {data_dir / 'val'}")

    paths = [row["path"] for row in pooled]
    labels = [row["label"] for row in pooled]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )

    rows = [
        {"path": path, "label": label, "split": "train"}
        for path, label in zip(train_paths, train_labels)
    ]
    rows += [
        {"path": path, "label": label, "split": "val"}
        for path, label in zip(val_paths, val_labels)
    ]

    for row in collect_images(data_dir, ("test",)):
        rows.append({"path": row["path"], "label": row["label"], "split": "test"})

    frame = pd.DataFrame(rows, columns=["path", "label", "split"])
    frame = frame.sort_values(["split", "label", "path"]).reset_index(drop=True)
    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(splits_csv, index=False)

    summary = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    summary["total"] = summary.sum(axis=1)
    print(f"Wrote {splits_csv.relative_to(ROOT)} ({len(frame)} rows)")
    print(summary.to_string())


if __name__ == "__main__":
    main()
