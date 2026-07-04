"""Centered-kernel-alignment helpers for encoder layer selection."""

from __future__ import annotations

import torch


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    if features.dim() < 2:
        raise ValueError("features must have at least two dimensions.")
    return features.reshape(features.shape[0], -1).to(dtype=torch.float64)


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-12) -> float:
    """Return linear CKA similarity between two sample-aligned feature matrices."""

    x_flat = _flatten_features(x)
    y_flat = _flatten_features(y)
    if x_flat.shape[0] != y_flat.shape[0]:
        raise ValueError("x and y must have the same number of samples for CKA.")
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    y_centered = y_flat - y_flat.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(x_centered.T @ y_centered, ord="fro") ** 2
    x_norm = torch.linalg.matrix_norm(x_centered.T @ x_centered, ord="fro")
    y_norm = torch.linalg.matrix_norm(y_centered.T @ y_centered, ord="fro")
    denom = x_norm * y_norm
    if float(denom) <= eps:
        return 0.0
    return float((cross / denom).clamp(min=0.0, max=1.0).item())


def rank_layer_pairs(
    vision_layers: dict[int, torch.Tensor],
    text_layers: dict[int, torch.Tensor],
) -> list[dict[str, float | int]]:
    """Rank all vision/text layer pairs by linear CKA in descending order."""

    rows: list[dict[str, float | int]] = []
    for vision_layer, vision_features in vision_layers.items():
        for text_layer, text_features in text_layers.items():
            rows.append(
                {
                    "vision_layer": int(vision_layer),
                    "text_layer": int(text_layer),
                    "cka": linear_cka(vision_features, text_features),
                }
            )
    return sorted(rows, key=lambda row: float(row["cka"]), reverse=True)
