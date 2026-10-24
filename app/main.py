"""Streamlit demo: pneumonia classification with Grad-CAM explanations."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLASSES, build_transforms, load_config  # noqa: E402
from src.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from src.model import build_model  # noqa: E402

CHECKPOINT = ROOT / "models" / "resnet18_xray.pth"


@st.cache_resource(show_spinner="Loading model...")
def load_bundle(checkpoint: str):
    """Load the classifier and Grad-CAM helper once per server process."""
    config = load_config()
    model = build_model(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model, GradCAM(model), config


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into an RGB ``uint8`` array."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded file as an image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(
        page_title="Chest X-ray pneumonia classifier",
        layout="wide",
    )

    st.title("Chest X-ray pneumonia classifier")
    st.caption("ResNet18 classifier with Grad-CAM explanations")

    with st.sidebar:
        st.header("About")
        st.markdown(
            "This demo classifies a chest X-ray as **NORMAL** or "
            "**PNEUMONIA** and highlights the regions that drove the "
            "prediction using Grad-CAM."
        )
        st.markdown(
            "1. Upload a JPG or PNG chest X-ray.\n"
            "2. The model scores both classes.\n"
            "3. Grad-CAM shows where it looked."
        )
        st.warning(
            "Research demo only, not a medical device. Do not use for "
            "clinical decisions.",
            title="Disclaimer",
        )

    if not CHECKPOINT.is_file():
        st.error(
            f"Checkpoint not found at `{CHECKPOINT.relative_to(ROOT)}`. "
            "Train a model first.",
            title="Model not available",
        )
        st.stop()

    model, gradcam, config = load_bundle(str(CHECKPOINT))
    transform = build_transforms(int(config["common"]["img_size"]), train=False)

    uploaded = st.file_uploader(
        "Upload a chest X-ray",
        type=["jpg", "jpeg", "png"],
        help="JPG or PNG image of a frontal chest X-ray.",
    )

    if uploaded is None:
        st.info("Upload an X-ray to get a prediction and Grad-CAM heatmap.")
        return

    try:
        image = decode_image(uploaded.getvalue())
    except ValueError as error:
        st.error(str(error), title="Unsupported image")
        st.stop()

    tensor = transform(image=image)["image"].unsqueeze(0)
    threshold = float(config["common"].get("decision_threshold", 0.5))

    with st.spinner("Running the model..."):
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0].numpy()
        predicted_index = int(probabilities[1] >= threshold)
        heatmaps, _ = gradcam.generate(tensor, target_class=predicted_index)

    overlay = overlay_heatmap(image, heatmaps[0])
    pneumonia_score = float(probabilities[1])

    if predicted_index == 1:
        st.error(
            f"Pneumonia probability {pneumonia_score:.1%} "
            f"(threshold {threshold:.0%})",
            title="Pneumonia detected",
            icon=":material/warning:",
        )
    else:
        st.success(
            f"Pneumonia probability {pneumonia_score:.1%} "
            f"(threshold {threshold:.0%})",
            title="No pneumonia detected",
            icon=":material/check_circle:",
        )

    normal_col, pneumonia_col = st.columns(2)
    normal_col.metric("Normal", f"{probabilities[0]:.1%}", border=True)
    pneumonia_col.metric("Pneumonia", f"{probabilities[1]:.1%}", border=True)
    st.caption(
        f"Pneumonia is flagged when the score is at or above {threshold:.0%}. "
        "Change the decision threshold in `configs/config.yaml`."
    )

    original_col, heatmap_col = st.columns(2)
    original_col.image(
        image,
        caption="Original X-ray",
        width="stretch",
        alt="Uploaded chest X-ray",
    )
    heatmap_col.image(
        overlay,
        caption="Grad-CAM",
        width="stretch",
        alt="Grad-CAM heatmap overlaid on the chest X-ray",
    )
    st.caption(
        "Warmer regions in the Grad-CAM view contributed more to the "
        "predicted class."
    )


main()
