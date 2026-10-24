"""Tests for the chest X-ray model builder."""

from __future__ import annotations

import torch

from src.model import build_model, get_gradcam_layer


def test_model_outputs_two_classes() -> None:
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))

    assert outputs.shape == (2, 2)
    assert model.fc.out_features == 2


def test_gradcam_layer_is_last_block() -> None:
    model = build_model(num_classes=2, pretrained=False)
    assert get_gradcam_layer(model) is model.layer4
