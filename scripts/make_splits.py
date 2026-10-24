"""Build data/splits.csv from the raw chest X-ray dataset.

Merges the tiny Kaggle ``val`` split into ``train`` and performs a seeded,
stratified re-split so the validation set is large enough to be meaningful.
The ``test`` split is copied through untouched.
"""

from __future__ import annotations

import hashlib
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


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    test_rows = collect_images(data_dir, ("test",))

    for row in pooled + test_rows:
        row["md5"] = file_hash(ROOT / row["path"])

    pool = pd.DataFrame(pooled).drop_duplicates("md5", keep="first")
    test = pd.DataFrame(test_rows).drop_duplicates("md5", keep="first")
    duplicate_pool = len(pooled) - len(pool)
    duplicate_test = len(test_rows) - len(test)

    test_hashes = set(test["md5"])
    leaked = int(pool["md5"].isin(test_hashes).sum())
    pool = pool[~pool["md5"].isin(test_hashes)]

    paths = pool["path"].tolist()
    labels = pool["label"].tolist()

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

    for _, row in test.iterrows():
        rows.append({"path": row["path"], "label": row["label"], "split": "test"})

    frame = pd.DataFrame(rows, columns=["path", "label", "split"])
    frame = frame.sort_values(["split", "label", "path"]).reset_index(drop=True)
    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(splits_csv, index=False)

    summary = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    summary["total"] = summary.sum(axis=1)
    print(f"Dropped {duplicate_pool} duplicate train/val images and {duplicate_test} duplicate test images")
    print(f"Dropped {leaked} train/val images that duplicate a test image")
    print(f"Wrote {splits_csv.relative_to(ROOT)} ({len(frame)} rows)")
    print(summary.to_string())


if __name__ == "__main__":
    main()
