"""Build leak-free data/splits.csv from the raw chest X-ray dataset.

Images are grouped by patient identifier and by perceptual hash before
splitting, so the validation set cannot share a patient or a near-duplicate
with training. The Kaggle ``val`` folder is merged into the pool and the
``test`` folder is kept as an external split; any pooled image that duplicates
a test image is dropped from the pool.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
CLASSES = ("NORMAL", "PNEUMONIA")
# A 16x16 difference hash (255 bits) separates resized/recompressed copies
# (distance <= 4) from unrelated X-rays (distance >= 31) with a wide margin.
HASH_GRID = 16
NEAR_DUPLICATE_DISTANCE = 6


def load_config() -> dict:
    with (ROOT / "configs" / "config.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["common"]


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> int | None:
    """Return a difference hash (dHash) for the image at ``path``."""
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if image is None:
        return None
    resized = cv2.resize(
        image, (HASH_GRID + 1, HASH_GRID), interpolation=cv2.INTER_AREA
    )
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def patient_id(path: Path) -> str:
    """Derive a patient/study identifier from the image filename."""
    stem = path.stem
    match = re.search(r"person\d+", stem, flags=re.IGNORECASE)
    if match:
        return match.group(0).lower()
    match = re.search(r"(?:normal2?-)?im-\d+", stem, flags=re.IGNORECASE)
    if match:
        return match.group(0).lower()
    return stem.lower()


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


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def count_duplicates(frame: pd.DataFrame) -> int:
    return len(frame) - frame["md5"].nunique()


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
        path = ROOT / row["path"]
        row["md5"] = file_hash(path)
        row["phash"] = perceptual_hash(path)

    pool = pd.DataFrame(pooled).drop_duplicates("md5", keep="first").reset_index(drop=True)
    test = pd.DataFrame(test_rows).drop_duplicates("md5", keep="first").reset_index(drop=True)
    exact_pool_duplicates = count_duplicates(pd.DataFrame(pooled))
    exact_test_duplicates = count_duplicates(pd.DataFrame(test_rows))

    test_md5 = set(test["md5"])
    test_hashes = [value for value in test["phash"] if value is not None]
    test_groups = {
        patient_id(ROOT / row.path) for row in test.itertuples(index=False)
    }

    pool = pool[~pool["md5"].isin(test_md5)].reset_index(drop=True)

    before_patient_filter = len(pool)
    pool = pool[
        ~pool["path"].map(lambda value: patient_id(ROOT / value)).isin(test_groups)
    ].reset_index(drop=True)
    patient_test_removed = before_patient_filter - len(pool)

    keep = []
    near_test = 0
    for row in pool.itertuples(index=False):
        if row.phash is None:
            keep.append(True)
            continue
        duplicate = any(
            (row.phash ^ other).bit_count() <= NEAR_DUPLICATE_DISTANCE
            for other in test_hashes
        )
        near_test += int(duplicate)
        keep.append(not duplicate)
    pool = pool[keep].reset_index(drop=True)

    uf = UnionFind(len(pool))
    processed_groups: dict[str, int] = {}
    for index, row in enumerate(pool.itertuples(index=False)):
        group = patient_id(ROOT / row.path)
        if group in processed_groups:
            uf.union(index, processed_groups[group])
        else:
            processed_groups[group] = index

    hashes = pool["phash"].tolist()
    near_pairs = 0
    for first in range(len(pool)):
        if hashes[first] is None:
            continue
        for second in range(first + 1, len(pool)):
            if hashes[second] is None:
                continue
            if (hashes[first] ^ hashes[second]).bit_count() <= NEAR_DUPLICATE_DISTANCE:
                uf.union(first, second)
                near_pairs += 1

    pool["group"] = [f"group-{uf.find(index)}" for index in range(len(pool))]
    group_purity = pool.groupby("group")["label"].nunique()
    mixed_groups = int((group_purity > 1).sum())

    n_splits = max(2, round(1.0 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_index, val_index = next(
        splitter.split(pool, pool["label"], groups=pool["group"])
    )

    rows = [
        {"path": row.path, "label": row.label, "split": "train"}
        for row in pool.iloc[train_index].itertuples(index=False)
    ]
    rows += [
        {"path": row.path, "label": row.label, "split": "val"}
        for row in pool.iloc[val_index].itertuples(index=False)
    ]
    for row in test.itertuples(index=False):
        rows.append({"path": row.path, "label": row.label, "split": "test"})

    frame = pd.DataFrame(rows, columns=["path", "label", "split"])
    frame = frame.sort_values(["split", "label", "path"]).reset_index(drop=True)
    splits_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(splits_csv, index=False)

    split_groups = frame.assign(
        group=frame["path"].map(lambda value: patient_id(ROOT / value))
    ).groupby("split")["group"].apply(set)
    overlap_train_val = split_groups["train"] & split_groups["val"]
    overlap_test = (split_groups["train"] | split_groups["val"]) & split_groups["test"]

    summary = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    summary["total"] = summary.sum(axis=1)
    print(f"Dropped {exact_pool_duplicates} exact duplicate train/val images")
    print(f"Dropped {exact_test_duplicates} exact duplicate test images")
    print(f"Dropped {patient_test_removed} train/val images sharing a patient with test")
    print(f"Dropped {near_test} train/val images near-duplicating a test image")
    print(f"Linked {near_pairs} near-duplicate train/val pairs")
    print(f"{pool['group'].nunique()} groups across {len(pool)} train/val images")
    if mixed_groups:
        print(f"WARNING: {mixed_groups} groups contain more than one label")
    print(f"Patient-id overlap train/val: {len(overlap_train_val)}")
    print(f"Patient-id overlap with test: {len(overlap_test)}")
    if overlap_train_val or overlap_test:
        print("WARNING: group overlap remains; inspect the split")
    print(f"\nWrote {splits_csv.relative_to(ROOT)} ({len(frame)} rows)")
    print(summary.to_string())


if __name__ == "__main__":
    main()
