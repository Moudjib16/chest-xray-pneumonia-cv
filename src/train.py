"""Train a ResNet18 on the chest X-ray dataset."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import build_dataloaders, load_config  # noqa: E402
from src.model import build_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_cpu(threads: int) -> None:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(2)


def compute_class_weights(dataset) -> torch.Tensor:
    counts = np.bincount(np.asarray(dataset.labels), minlength=2)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 on chest X-rays")
    parser.add_argument("--profile", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "resnet18_xray.pth")
    args = parser.parse_args()

    config = load_config()
    common = config["common"]
    profile = config[args.profile]
    epochs = args.epochs or profile["epochs"]

    set_seed(int(common["seed"]))
    configure_cpu(int(profile["cpu_threads"]))

    train_loader = build_dataloaders(profile=args.profile, split="train", config=config)
    val_loader = build_dataloaders(profile=args.profile, split="val", config=config)

    model = build_model(num_classes=2, pretrained=True)

    if common.get("use_class_weights", False):
        weights = compute_class_weights(train_loader.dataset)
        print(f"class weights: NORMAL={weights[0]:.3f} PNEUMONIA={weights[1]:.3f}")
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for name, p in model.named_parameters() if not name.startswith("fc.")],
                "lr": common["lr_backbone"],
            },
            {"params": model.fc.parameters(), "lr": common["lr_head"]},
        ],
        weight_decay=common["weight_decay"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        elapsed = time.perf_counter() - start

        print(
            f"epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  saved best weights (val_acc={val_acc:.4f}) -> {args.output}")

    print(f"training complete: best val acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
