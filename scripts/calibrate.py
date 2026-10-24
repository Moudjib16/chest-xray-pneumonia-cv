"""Choose a decision threshold for the chest X-ray classifier.

Sweeps thresholds on a validation split and reports the operating point that
maximises Youden's J statistic, the point with the best accuracy, and the most
specific threshold that still meets a target sensitivity. Writes the numbers to
``reports/calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import collect_predictions  # noqa: E402
from src.dataset import CLASSES, build_dataloaders, load_config  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import configure_cpu  # noqa: E402


def sensitivity_specificity(probabilities, labels, threshold: float) -> tuple[float, float, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(labels)
    return float(sensitivity), float(specificity), float(accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick a decision threshold")
    parser.add_argument("--profile", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--target-sensitivity", type=float, default=0.98)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "resnet18_xray.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "calibration.json",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    config = load_config()
    configure_cpu(int(config[args.profile]["cpu_threads"]))

    loader = build_dataloaders(profile=args.profile, split=args.split, config=config)
    model = build_model(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    probabilities, labels = collect_predictions(model, loader, torch.device("cpu"))
    pneumonia_scores = probabilities[:, 1]

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, pneumonia_scores
    )
    youden_index = int(np.argmax(true_positive_rate - false_positive_rate))
    threshold_youden = float(thresholds[youden_index])

    grid = np.linspace(0.05, 0.95, 91)
    sweep = [sensitivity_specificity(pneumonia_scores, labels, float(t)) for t in grid]
    best_index = int(np.argmax([item[2] for item in sweep]))
    threshold_best_accuracy = float(grid[best_index])

    meeting_target = [
        float(threshold)
        for threshold in grid
        if sensitivity_specificity(pneumonia_scores, labels, float(threshold))[0]
        >= args.target_sensitivity
    ]
    threshold_target = max(meeting_target) if meeting_target else float(grid[0])

    report = {
        "split": args.split,
        "n_samples": int(labels.size),
        "target_sensitivity": args.target_sensitivity,
        "default_threshold": 0.5,
        "threshold_youden": threshold_youden,
        "threshold_best_accuracy": threshold_best_accuracy,
        "threshold_target_sensitivity": threshold_target,
    }
    for name, threshold in (
        ("default", 0.5),
        ("youden", threshold_youden),
        ("best_accuracy", threshold_best_accuracy),
        ("target_sensitivity", threshold_target),
    ):
        sensitivity, specificity, accuracy = sensitivity_specificity(
            pneumonia_scores, labels, threshold
        )
        report[f"metrics_{name}"] = {
            "threshold": threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy,
        }
        print(
            f"{name:>17}: threshold={threshold:.3f} "
            f"sens={sensitivity:.3f} spec={specificity:.3f} acc={accuracy:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nsaved calibration -> {args.output}")


if __name__ == "__main__":
    main()
