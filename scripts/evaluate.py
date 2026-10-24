"""Evaluate a trained checkpoint on the chest X-ray splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLASSES, build_dataloaders, load_config  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import configure_cpu  # noqa: E402


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device):
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for images, batch_labels in loader:
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(batch_labels.numpy())

    return np.concatenate(probabilities), np.concatenate(labels)


def compute_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    predictions = (probabilities[:, 1] >= threshold).astype(int)
    pneumonia_scores = probabilities[:, 1]

    return {
        "threshold": float(threshold),
        "n_samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, pneumonia_scores)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=list(CLASSES),
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix_figure(matrix: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(4.5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    axis.set_yticks(range(len(CLASSES)), CLASSES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a chest X-ray checkpoint")
    parser.add_argument("--profile", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="pneumonia probability above which the sample is positive",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "resnet18_xray.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (default: reports/<split>_metrics.json)",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="optional path for a confusion matrix PNG",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    config = load_config()
    configure_cpu(int(config[args.profile]["cpu_threads"]))

    loader = build_dataloaders(
        profile=args.profile, split=args.split, config=config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    probabilities, labels = collect_predictions(model, loader, device)
    metrics = compute_metrics(probabilities, labels, threshold=args.threshold)

    print(
        f"split: {args.split} | samples: {metrics['n_samples']} | "
        f"threshold: {metrics['threshold']:.3f} | device: {device}"
    )
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"roc_auc:   {metrics['roc_auc']:.4f}")
    print("\nconfusion matrix (rows=actual, cols=predicted):")
    print(np.asarray(metrics["confusion_matrix"]))

    output = args.output or ROOT / "reports" / f"{args.split}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nsaved metrics -> {output}")

    if args.figure:
        save_confusion_matrix_figure(
            np.asarray(metrics["confusion_matrix"]), args.figure
        )
        print(f"saved confusion matrix -> {args.figure}")


if __name__ == "__main__":
    main()
