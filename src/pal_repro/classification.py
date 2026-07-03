"""Zero-shot classification helpers for PAL paper reproduction."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch


def normalize_class_name(name: str) -> str:
    """Normalize dataset class labels into prompt-friendly text."""

    text = str(name).replace("_", " ").replace("-", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def build_class_prompts(
    class_names: Iterable[str],
    template: str = "a photo of {class_name}",
) -> list[str]:
    """Build class prompts from raw dataset class names."""

    return [template.format(class_name=normalize_class_name(name)) for name in class_names]


def classification_topk(
    similarity: torch.Tensor,
    labels: torch.Tensor,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Return top-k classification accuracies as percentages."""

    if similarity.dim() != 2:
        raise ValueError("similarity must be a 2D tensor shaped (N, C).")
    labels = labels.detach().cpu().long().flatten()
    if labels.shape[0] != similarity.shape[0]:
        raise ValueError("labels must contain one target per similarity row.")
    metrics: dict[str, float] = {}
    for k in ks:
        if k <= 0:
            raise ValueError("k must be positive.")
        kk = min(k, similarity.shape[1])
        topk = similarity.detach().cpu().topk(k=kk, dim=1).indices
        metrics[f"top{k}"] = topk.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
    return metrics
