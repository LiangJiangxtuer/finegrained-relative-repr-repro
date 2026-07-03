"""Tensor data utilities for PAL training on pre-extracted tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TokenTensors:
    """In-memory token tensors for paired image/text examples."""

    image_tokens: torch.Tensor
    text_tokens: torch.Tensor
    text_mask: torch.Tensor

    def __post_init__(self) -> None:
        n = self.image_tokens.shape[0]
        if self.image_tokens.dim() != 3:
            raise ValueError("image_tokens must have shape (N, T_v, D_v).")
        if self.text_tokens.dim() != 3:
            raise ValueError("text_tokens must have shape (N, T_l, D_l).")
        if self.text_mask.dim() != 2:
            raise ValueError("text_mask must have shape (N, T_l).")
        if self.text_tokens.shape[:2] != self.text_mask.shape:
            raise ValueError(
                "text_mask must match text token batch/token dimensions, "
                f"got {tuple(self.text_mask.shape)} vs {tuple(self.text_tokens.shape[:2])}."
            )
        if self.text_tokens.shape[0] != n or self.text_mask.shape[0] != n:
            raise ValueError("image/text/mask tensors must share the same N.")

    @property
    def num_samples(self) -> int:
        return int(self.image_tokens.shape[0])

    @property
    def dim_img(self) -> int:
        return int(self.image_tokens.shape[-1])

    @property
    def dim_txt(self) -> int:
        return int(self.text_tokens.shape[-1])


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> torch.Tensor:
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=map_location)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(obj)!r}.")
    return obj


def load_token_tensors(
    data_dir: str | Path,
    image_name: str = "image_tokens.pt",
    text_name: str = "text_tokens.pt",
    mask_name: str = "text_mask.pt",
    map_location: str | torch.device = "cpu",
) -> TokenTensors:
    """Load image tokens, text tokens, and text mask from a directory."""

    root = Path(data_dir)
    image_path = root / image_name
    text_path = root / text_name
    mask_path = root / mask_name
    missing = [str(path) for path in (image_path, text_path, mask_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing token tensor file(s): " + ", ".join(missing))

    # Preserve storage dtype (usually fp16 for full paper-scale caches). Batches
    # are cast to fp32 only when moved to the training/eval device.
    image_tokens = _torch_load(image_path, map_location=map_location)
    text_tokens = _torch_load(text_path, map_location=map_location)
    text_mask = _torch_load(mask_path, map_location=map_location).bool()
    return TokenTensors(image_tokens=image_tokens, text_tokens=text_tokens, text_mask=text_mask)


def split_indices(
    num_samples: int,
    train_size: int | None = None,
    seed: int = 42,
    train_fraction: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic disjoint train/eval indices."""

    if num_samples < 2:
        raise ValueError("Need at least two samples for a train/eval split.")
    if train_size is None:
        if not (0.0 < train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1).")
        train_size = int(round(num_samples * train_fraction))
    if train_size <= 0 or train_size > num_samples:
        raise ValueError(
            f"train_size must be in [1, {num_samples}], got {train_size}."
        )

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_samples, generator=generator)
    return perm[:train_size], perm[train_size:]


class TokenTensorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset view over pre-extracted paired token tensors."""

    def __init__(self, tensors: TokenTensors, indices: torch.Tensor | None = None) -> None:
        self.tensors = tensors
        if indices is None:
            indices = torch.arange(tensors.num_samples)
        if indices.dim() != 1:
            raise ValueError("indices must be 1D.")
        self.indices = indices.long().cpu()

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        return (
            self.tensors.image_tokens[idx],
            self.tensors.text_tokens[idx],
            self.tensors.text_mask[idx],
        )
