"""Evaluation utilities for PAL profiles and paper tasks."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def recall_at_k(similarity: torch.Tensor, k: int = 1) -> float:
    """Return diagonal-positive recall@k as a fraction in ``[0, 1]``."""

    if similarity.dim() != 2:
        raise ValueError("similarity must be a 2D matrix.")
    if similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square for one-to-one retrieval metrics.")
    if k <= 0:
        raise ValueError("k must be positive.")
    n = similarity.shape[0]
    k = min(k, n)
    topk = similarity.topk(k=k, dim=1).indices
    targets = torch.arange(n, device=similarity.device).unsqueeze(1)
    return topk.eq(targets).any(dim=1).float().mean().item()


def retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    percent: bool = False,
) -> dict[str, float]:
    """Compute one-positive image-to-text and text-to-image recall metrics."""

    if image_features.shape != text_features.shape:
        raise ValueError(
            "image_features and text_features must have same shape, "
            f"got {tuple(image_features.shape)} and {tuple(text_features.shape)}."
        )
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    similarity = image_features @ text_features.T
    scale = 100.0 if percent else 1.0
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"i2t_r{k}"] = recall_at_k(similarity, k=k) * scale
        metrics[f"t2i_r{k}"] = recall_at_k(similarity.T, k=k) * scale
    return metrics


def _normalize_ids(ids: Iterable[int] | torch.Tensor, expected_len: int, name: str) -> list[int]:
    if isinstance(ids, torch.Tensor):
        tensor_ids = ids.detach().cpu().flatten()
        values = [int(item) for item in tensor_ids.tolist()]
    else:
        values = [int(item) for item in ids]
    if len(values) != expected_len:
        raise ValueError(f"{name} length must be {expected_len}, got {len(values)}.")
    return values


def _multi_positive_recall_at_k(
    similarity: torch.Tensor,
    query_ids: list[int],
    candidate_ids: list[int],
    k: int,
) -> float:
    if similarity.shape != (len(query_ids), len(candidate_ids)):
        raise ValueError("similarity shape must match query_ids and candidate_ids lengths.")
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, similarity.shape[1])
    topk = similarity.topk(k=k, dim=1).indices.cpu()
    hits = 0
    for row, query_id in enumerate(query_ids):
        if any(candidate_ids[col] == query_id for col in topk[row].tolist()):
            hits += 1
    return hits / max(len(query_ids), 1)


def multicaption_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_ids: Iterable[int] | torch.Tensor,
    text_image_ids: Iterable[int] | torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
    percent: bool = False,
) -> dict[str, float]:
    """Compute retrieval recall when each image can have multiple positive captions.

    ``image_ids`` identifies each image row. ``text_image_ids`` identifies the
    source image for each text row. Image-to-text is correct when any caption
    from the query image appears in top-k; text-to-image is correct when the
    query caption's image appears in top-k image rows.
    """

    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError("image_features and text_features must be 2D.")
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError("image_features and text_features must share feature dimension.")
    image_id_list = _normalize_ids(image_ids, image_features.shape[0], "image_ids")
    text_image_id_list = _normalize_ids(text_image_ids, text_features.shape[0], "text_image_ids")
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    similarity = image_features @ text_features.T
    scale = 100.0 if percent else 1.0
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"i2t_r{k}"] = _multi_positive_recall_at_k(
            similarity,
            image_id_list,
            text_image_id_list,
            k=k,
        ) * scale
        metrics[f"t2i_r{k}"] = _multi_positive_recall_at_k(
            similarity.T,
            text_image_id_list,
            image_id_list,
            k=k,
        ) * scale
    metrics["mean_recall"] = sum(metrics.values()) / len(metrics)
    return metrics


@torch.no_grad()
def encode_images_batched(
    model: torch.nn.Module,
    image_tokens: torch.Tensor,
    batch_size: int = 128,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode image tokens with ``model.encode_image`` in batches."""

    model.eval()
    resolved = _resolve_model_device(model, device)
    outputs: list[torch.Tensor] = []
    for start in range(0, image_tokens.shape[0], batch_size):
        batch = image_tokens[start:start + batch_size].to(resolved, dtype=torch.float32)
        outputs.append(model.encode_image(batch).detach().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_texts_batched(
    model: torch.nn.Module,
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor | None = None,
    batch_size: int = 128,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode text tokens with ``model.encode_text`` in batches."""

    model.eval()
    resolved = _resolve_model_device(model, device)
    outputs: list[torch.Tensor] = []
    for start in range(0, text_tokens.shape[0], batch_size):
        end = min(start + batch_size, text_tokens.shape[0])
        batch = text_tokens[start:end].to(resolved, dtype=torch.float32)
        mask = None if text_mask is None else text_mask[start:end].to(resolved)
        outputs.append(model.encode_text(batch, mask).detach().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def evaluate_retrieval_model(
    model: torch.nn.Module,
    image_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor | None = None,
    ks: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 128,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Evaluate paired image-text retrieval and return paper-style percentages."""

    if image_tokens.shape[0] != text_tokens.shape[0]:
        raise ValueError("retrieval expects one text per image in matching order.")
    image_features = encode_images_batched(model, image_tokens, batch_size=batch_size, device=device)
    text_features = encode_texts_batched(model, text_tokens, text_mask, batch_size=batch_size, device=device)
    metrics = retrieval_metrics(image_features, text_features, ks=ks, percent=True)
    metrics["mean_recall"] = sum(metrics.values()) / len(metrics)
    return metrics


@torch.no_grad()
def evaluate_zero_shot_classification(
    model: torch.nn.Module,
    image_tokens: torch.Tensor,
    class_text_tokens: torch.Tensor,
    class_text_mask: torch.Tensor | None,
    labels: torch.Tensor,
    batch_size: int = 128,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Zero-shot classification using PAL image profiles vs class text profiles.

    Returns top-1/top-5 percentages, matching the paper's top-1 reporting
    while retaining top-5 for debugging.
    """

    if labels.dim() != 1 or labels.shape[0] != image_tokens.shape[0]:
        raise ValueError("labels must be a 1D tensor aligned with image_tokens.")
    image_features = encode_images_batched(model, image_tokens, batch_size=batch_size, device=device)
    class_features = encode_texts_batched(
        model,
        class_text_tokens,
        class_text_mask,
        batch_size=batch_size,
        device=device,
    )
    similarity = F.normalize(image_features, dim=-1) @ F.normalize(class_features, dim=-1).T
    labels = labels.cpu().long()
    pred1 = similarity.argmax(dim=1)
    top1 = pred1.eq(labels).float().mean().item() * 100.0
    k = min(5, similarity.shape[1])
    topk = similarity.topk(k=k, dim=1).indices
    top5 = topk.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
    return {"top1": top1, "top5": top5}


def foreground_miou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_ids: Iterable[int],
    background_id: int = 0,
    ignore_index: int | None = None,
) -> float:
    """Compute foreground mIoU percentage for segmentation masks.

    Classes absent from both prediction and target are skipped. ``background_id``
    is ignored even if present in ``class_ids``.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target masks must have the same shape.")
    pred = prediction.cpu()
    tgt = target.cpu()
    valid = torch.ones_like(tgt, dtype=torch.bool)
    if ignore_index is not None:
        valid = tgt != ignore_index
    ious: list[float] = []
    for class_id in class_ids:
        if class_id == background_id:
            continue
        pred_c = (pred == class_id) & valid
        tgt_c = (tgt == class_id) & valid
        union = (pred_c | tgt_c).sum().item()
        if union == 0:
            continue
        inter = (pred_c & tgt_c).sum().item()
        ious.append(inter / union)
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious) * 100.0)


def _resolve_model_device(
    model: torch.nn.Module,
    device: torch.device | str | None,
) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
        model.to(resolved)
        return resolved
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
