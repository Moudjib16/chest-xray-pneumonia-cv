"""Render a Grad-CAM sanity-check gallery from a split.

For one example per outcome (correct normal, correct pneumonia, false positive,
false negative) the script shows the original X-ray and its Grad-CAM overlay so
you can check the model attends to lung regions rather than artefacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLASSES, build_dataloaders, build_transforms, load_config  # noqa: E402
from src.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from src.model import build_model  # noqa: E402


def select_categories(probabilities, labels, threshold: float) -> dict[str, np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    categories = {
        "normal (correct)": np.where((labels == 0) & (predictions == 0))[0],
        "pneumonia (correct)": np.where((labels == 1) & (predictions == 1))[0],
        "false positive": np.where((labels == 0) & (predictions == 1))[0],
        "false negative": np.where((labels == 1) & (predictions == 0))[0],
    }
    ordered = {}
    for name, indices in categories.items():
        if indices.size == 0:
            ordered[name] = indices
            continue
        scores = probabilities[indices]
        order = np.argsort(scores)[::-1] if name == "false positive" else np.argsort(scores)
        ordered[name] = indices[order]
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Grad-CAM gallery")
    parser.add_argument("--profile", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "models" / "resnet18_xray.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "figures" / "gradcam_gallery.png",
    )
    args = parser.parse_args()

    config = load_config()
    threshold = float(config["common"].get("decision_threshold", 0.5))

    loader = build_dataloaders(profile=args.profile, split=args.split, config=config)
    paths = loader.dataset.paths
    labels = np.asarray(loader.dataset.labels)

    model = build_model(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    gradcam = GradCAM(model)

    probabilities = []
    with torch.no_grad():
        for images, _ in loader:
            probabilities.append(torch.softmax(model(images), dim=1)[:, 1].numpy())
    probabilities = np.concatenate(probabilities)

    categories = select_categories(probabilities, labels, threshold)
    transform = build_transforms(int(config["common"]["img_size"]), train=False)

    names = list(categories)
    fig, axes = plt.subplots(2, len(names), figsize=(4 * len(names), 8))
    for column, name in enumerate(names):
        indices = categories[name]
        if indices.size == 0:
            axes[0, column].axis("off")
            axes[1, column].axis("off")
            axes[0, column].set_title(f"{name}\n(none)")
            continue

        index = int(indices[0])
        path = paths[index]
        image = cv2.cvtColor(cv2.imread(str(ROOT / path)), cv2.COLOR_BGR2RGB)
        tensor = transform(image=image)["image"].unsqueeze(0)
        predicted_index = int(probabilities[index] >= threshold)
        heatmaps, _ = gradcam.generate(tensor, target_class=predicted_index)
        overlay = overlay_heatmap(image, heatmaps[0])

        axes[0, column].imshow(image)
        axes[0, column].set_title(f"{name}\np(pneumonia)={probabilities[index]:.2f}")
        axes[1, column].imshow(overlay)
        axes[1, column].set_title("Grad-CAM")
        for row in range(2):
            axes[row, column].axis("off")

    fig.suptitle(f"Grad-CAM sanity check — {args.split} split", fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(f"saved {args.output}")
    gradcam.remove_hooks()


if __name__ == "__main__":
    main()
