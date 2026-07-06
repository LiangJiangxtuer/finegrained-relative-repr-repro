"""Zero-shot classification helpers for PAL paper reproduction."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch

DEFAULT_PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of {class_name}",
    "a close-up photo of {class_name}",
    "a cropped photo of {class_name}",
    "a clean photo of {class_name}",
)


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


def build_class_prompt_groups(
    class_names: Iterable[str],
    templates: Iterable[str] = DEFAULT_PROMPT_TEMPLATES,
) -> list[list[str]]:
    """Build one fixed prompt ensemble per class."""

    template_list = list(templates)
    if not template_list:
        raise ValueError("templates must contain at least one prompt template.")
    return [
        [template.format(class_name=normalize_class_name(name)) for template in template_list]
        for name in class_names
    ]


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


def per_class_accuracy(
    similarity: torch.Tensor,
    labels: torch.Tensor,
    class_names: Iterable[str],
) -> dict[str, float | None]:
    """Return top-1 accuracy percentage for every class."""

    if similarity.dim() != 2:
        raise ValueError("similarity must be a 2D tensor shaped (N, C).")
    labels = labels.detach().cpu().long().flatten()
    if labels.shape[0] != similarity.shape[0]:
        raise ValueError("labels must contain one target per similarity row.")
    names = [normalize_class_name(name) for name in class_names]
    if len(names) != similarity.shape[1]:
        raise ValueError("class_names length must match similarity columns.")

    pred = similarity.detach().cpu().argmax(dim=1)
    rows: dict[str, float | None] = {}
    for idx, name in enumerate(names):
        mask = labels == idx
        if not mask.any():
            rows[name] = None
        else:
            rows[name] = pred[mask].eq(labels[mask]).float().mean().item() * 100.0
    return rows
