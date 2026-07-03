"""Loss functions for PAL reproduction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_info_nce_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric image-text InfoNCE loss.

    The diagonal of the batch similarity matrix is treated as the positive
    pair set. Inputs should already be L2-normalized PAL profiles.

    Args:
        image_features: Image features shaped ``(B,K)``.
        text_features: Text features shaped ``(B,K)``.
        temperature: Contrastive temperature ``tau``.

    Returns:
        Scalar loss averaging image-to-text and text-to-image cross entropy.
    """

    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError("image_features and text_features must be 2D tensors.")
    if image_features.shape != text_features.shape:
        raise ValueError(
            "image_features and text_features must have identical shape, "
            f"got {tuple(image_features.shape)} and {tuple(text_features.shape)}."
        )
    if image_features.shape[0] <= 0:
        raise ValueError("batch size must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    logits = image_features @ text_features.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0


# Compatibility alias for reference-code naming.
info_nce_loss = symmetric_info_nce_loss
