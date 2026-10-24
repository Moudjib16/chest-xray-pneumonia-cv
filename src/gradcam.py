"""Grad-CAM visualisation utilities for the chest X-ray classifier."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.model import get_gradcam_layer


class GradCAM:
    """Generate Grad-CAM heatmaps for a convolutional classifier.

    The default target layer is ``model.layer4`` for ResNet-style networks.
    Activations from that layer are weighted by the spatially averaged
    gradients of a chosen class score, passed through ReLU, and upsampled to
    the input resolution.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or get_gradcam_layer(model)
        self.activations: torch.Tensor | None = None
        self._handle = self.target_layer.register_forward_hook(self._capture_activations)

    def _capture_activations(self, _module, _inputs, output) -> None:
        self.activations = output

    def generate(
        self,
        images: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(heatmaps, targets)`` for a batch of images.

        ``images`` is a ``(N, 3, H, W)`` tensor or a single ``(3, H, W)``
        tensor. ``heatmaps`` is a float32 array of shape ``(N, H, W)`` with
        values in ``[0, 1]`` and ``targets`` holds the class index used for
        each heatmap (the predicted class when ``target_class`` is ``None``).
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError("images must have shape (N, 3, H, W) or (3, H, W)")

        device = next(self.model.parameters()).device
        images = images.to(device)

        was_training = self.model.training
        self.model.eval()
        self.activations = None
        try:
            with torch.enable_grad():
                logits = self.model(images)
                if target_class is None:
                    targets = logits.argmax(dim=1)
                else:
                    targets = torch.full(
                        (images.size(0),),
                        int(target_class),
                        dtype=torch.long,
                        device=device,
                    )
                scores = logits.gather(1, targets.unsqueeze(1)).sum()
                if self.activations is None:
                    raise RuntimeError("forward hook did not capture any activations")
                gradients = torch.autograd.grad(scores, self.activations)[0]

            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cams = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
            cams = F.interpolate(
                cams,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            cams = cams.squeeze(1).detach().cpu().numpy()
        finally:
            self.activations = None
            self.model.train(was_training)

        heatmaps = np.stack([_normalise(cam) for cam in cams]).astype(np.float32)
        return heatmaps, targets.detach().cpu().numpy()

    def remove_hooks(self) -> None:
        """Detach the forward hook from the target layer."""
        self._handle.remove()


def _normalise(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    peak = cam.max()
    if peak > 0:
        cam = cam / peak
    return cam


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape (H, W), (H, W, 3)")
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return array


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a ``[0, 1]`` heatmap over an image and return RGB ``uint8``.

    ``image`` may be grayscale or RGB, either ``uint8`` or float in ``[0, 1]``.
    ``heatmap`` is a 2-D array such as one sample from ``GradCAM.generate``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    rgb = _to_uint8_rgb(image)
    height, width = rgb.shape[:2]

    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.ndim != 2:
        raise ValueError("heatmap must be a 2-D array")
    heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_LINEAR)
    heat = np.clip(heat, 0.0, 1.0)

    coloured = cv2.applyColorMap(np.uint8(heat * 255), colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1.0 - alpha, coloured, alpha, 0)
