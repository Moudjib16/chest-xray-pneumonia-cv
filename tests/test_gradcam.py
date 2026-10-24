"""Tests for Grad-CAM heatmap generation and overlaying."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.gradcam import GradCAM, overlay_heatmap
from src.model import build_model

IMAGE_SIZE = 64


@pytest.fixture(scope="module")
def model():
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def gradcam(model):
    cam = GradCAM(model)
    yield cam
    cam.remove_hooks()


def test_generate_returns_normalised_heatmaps(gradcam) -> None:
    images = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    heatmaps, targets = gradcam.generate(images)

    assert heatmaps.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert heatmaps.dtype == np.float32
    assert 0.0 <= heatmaps.min() <= heatmaps.max() <= 1.0
    assert targets.shape == (1,)
    assert int(targets[0]) in {0, 1}


def test_generate_accepts_single_image(gradcam) -> None:
    heatmaps, targets = gradcam.generate(torch.randn(3, IMAGE_SIZE, IMAGE_SIZE))
    assert heatmaps.shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert targets.tolist() == [0] or targets.tolist() == [1]


def test_generate_uses_requested_target(gradcam) -> None:
    images = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    heatmaps, targets = gradcam.generate(images, target_class=1)

    assert heatmaps.shape == (2, IMAGE_SIZE, IMAGE_SIZE)
    assert targets.tolist() == [1, 1]


def test_generate_requires_forward_hook(model) -> None:
    cam = GradCAM(model)
    cam.remove_hooks()

    with pytest.raises(RuntimeError, match="forward hook"):
        cam.generate(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))


def test_overlay_returns_rgb_uint8() -> None:
    image = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    heatmap = np.random.rand(8, 8).astype(np.float32)

    overlay = overlay_heatmap(image, heatmap)

    assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8


def test_overlay_accepts_float_grayscale() -> None:
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    heatmap = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    overlay = overlay_heatmap(image, heatmap, alpha=0.5)

    assert overlay.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert overlay.dtype == np.uint8


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_overlay_rejects_invalid_alpha(alpha: float) -> None:
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    with pytest.raises(ValueError, match="alpha"):
        overlay_heatmap(image, heatmap, alpha=alpha)


def test_overlay_rejects_non_2d_heatmap() -> None:
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="2-D"):
        overlay_heatmap(image, heatmap)
