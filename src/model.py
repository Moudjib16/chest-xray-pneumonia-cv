"""Model definition for chest X-ray classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Return a ResNet18 with its classifier replaced for ``num_classes``."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_gradcam_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional block used for Grad-CAM."""
    return model.layer4
