"""Analysis utilities for PAL learned anchors."""

from __future__ import annotations

import torch


def topk_anchor_sets(profiles: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Return top-k anchor indices per sample from unnormalized/normalized profiles."""

    if profiles.dim() != 2:
        raise ValueError("profiles must be shaped (N, K).")
    if k <= 0:
        raise ValueError("k must be positive.")
    return profiles.topk(k=min(k, profiles.shape[1]), dim=1).indices


def hard_overlap_score(image_topk: torch.Tensor, text_topk: torch.Tensor) -> float:
    """Mean |A∩B|/k for paired top-k anchor index tensors."""

    if image_topk.shape != text_topk.shape:
        raise ValueError("top-k tensors must have the same shape.")
    scores = []
    for img_row, txt_row in zip(image_topk.tolist(), text_topk.tolist(), strict=True):
        img_set = set(img_row)
        txt_set = set(txt_row)
        scores.append(len(img_set & txt_set) / max(len(img_set), 1))
    return float(sum(scores) / len(scores)) if scores else 0.0


def dice_overlap_score(image_topk: torch.Tensor, text_topk: torch.Tensor) -> float:
    """Mean Dice coefficient between paired top-k anchor sets."""

    if image_topk.shape != text_topk.shape:
        raise ValueError("top-k tensors must have the same shape.")
    scores = []
    for img_row, txt_row in zip(image_topk.tolist(), text_topk.tolist(), strict=True):
        img_set = set(img_row)
        txt_set = set(txt_row)
        denom = len(img_set) + len(txt_set)
        scores.append(0.0 if denom == 0 else 2 * len(img_set & txt_set) / denom)
    return float(sum(scores) / len(scores)) if scores else 0.0


def anchor_overlap_report(
    image_profiles: torch.Tensor,
    text_profiles: torch.Tensor,
    k: int = 5,
    mismatched_shift: int = 1,
) -> dict[str, float]:
    """Compute matched/mismatched top-k anchor overlap metrics.

    Mismatched pairs are built by cyclically shifting text top-k anchors.
    """

    if image_profiles.shape != text_profiles.shape:
        raise ValueError("image/text profiles must have the same shape.")
    if image_profiles.shape[0] < 2:
        raise ValueError("need at least two samples for mismatched overlap.")
    image_topk = topk_anchor_sets(image_profiles, k=k)
    text_topk = topk_anchor_sets(text_profiles, k=k)
    mismatched_text_topk = torch.roll(text_topk, shifts=mismatched_shift, dims=0)
    return {
        "matched_hard_overlap": hard_overlap_score(image_topk, text_topk),
        "mismatched_hard_overlap": hard_overlap_score(image_topk, mismatched_text_topk),
        "matched_dice": dice_overlap_score(image_topk, text_topk),
        "mismatched_dice": dice_overlap_score(image_topk, mismatched_text_topk),
    }
