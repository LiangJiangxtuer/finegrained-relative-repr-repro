"""Dense segmentation helpers for PAL paper reproduction."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

VOC_CLASS_NAMES: list[str] = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
]


def parse_pascal_context_labels(lines: Iterable[str]) -> list[tuple[int, str]]:
    """Parse Pascal Context ``labels.txt`` rows as ``(label_id, name)`` pairs."""

    labels: list[tuple[int, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line or ":" not in line:
            continue
        idx, name = line.split(":", 1)
        labels.append((int(idx.strip()), name.strip()))
    return sorted(labels, key=lambda item: item[0])


def parse_ade20k_object_info(lines: Iterable[str]) -> list[tuple[int, str]]:
    """Parse ADE20K ``objectInfo150.txt`` rows using the first class-name alias."""

    labels: list[tuple[int, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("Idx"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        class_id = int(parts[0].strip())
        class_name = parts[-1].split(",", 1)[0].strip()
        labels.append((class_id, class_name))
    return sorted(labels, key=lambda item: item[0])


class PascalContextSegmentationDataset:
    """Pascal Context trainval dataset backed by ``LabelMap`` .mat masks."""

    def __init__(self, root: str | Path, image_root: str | Path | None = None) -> None:
        self.root = Path(root)
        label_path = self.root / "labels.txt"
        self.labels = parse_pascal_context_labels(label_path.read_text(encoding="utf-8").splitlines())
        self.class_ids = [item[0] for item in self.labels]
        self.class_names = [item[1] for item in self.labels]
        self.mask_paths = sorted((self.root / "trainval").glob("*.mat"))
        self.image_root = Path(image_root) if image_root is not None else None

    def __len__(self) -> int:
        return len(self.mask_paths)

    def _image_path(self, stem: str) -> Path:
        candidates: list[Path] = []
        if self.image_root is not None:
            candidates.append(self.image_root / f"{stem}.jpg")
        segmentation_root = self.root.parents[1] if len(self.root.parents) > 1 else self.root
        candidates.extend(
            [
                self.root / "JPEGImages" / f"{stem}.jpg",
                segmentation_root / "voc2010/raw/VOCdevkit/VOC2010/JPEGImages" / f"{stem}.jpg",
                segmentation_root / "voc2012/VOCdevkit/VOC2012/JPEGImages" / f"{stem}.jpg",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve Pascal Context JPEG for {stem}")

    def __getitem__(self, index: int):
        from scipy.io import loadmat

        mask_path = self.mask_paths[index]
        stem = mask_path.stem
        image = Image.open(self._image_path(stem)).convert("RGB")
        label_map = loadmat(mask_path)["LabelMap"].astype(np.int32)
        return image, Image.fromarray(label_map, mode="I")


class ADE20KSegmentationDataset:
    """ADE20K validation dataset for 150-class foreground mIoU evaluation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        info_path = self.root / "objectInfo150.txt"
        self.labels = parse_ade20k_object_info(info_path.read_text(encoding="utf-8").splitlines())
        self.class_ids = [item[0] for item in self.labels]
        self.class_names = [item[1] for item in self.labels]
        self.image_paths = sorted((self.root / "images" / "validation").glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mask_path = self.root / "annotations" / "validation" / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing ADE20K mask: {mask_path}")
        return Image.open(image_path).convert("RGB"), Image.open(mask_path)


def segmentation_prompts(class_names: Iterable[str], template: str = "a photo of {class_name}") -> list[str]:
    return [template.format(class_name=name) for name in class_names]


def image_patch_profiles(model: torch.nn.Module, image_tokens: torch.Tensor) -> torch.Tensor:
    """Return normalized PAL-relative patch profiles shaped ``(B, P, K)``.

    The first DINOv2 token is the CLS token and is excluded for dense prediction.
    """

    if image_tokens.dim() != 3:
        raise ValueError("image_tokens must be shaped (B, T, D).")
    if image_tokens.shape[1] <= 1:
        raise ValueError("image_tokens must include CLS plus at least one patch token.")
    patches = image_tokens[:, 1:, :]
    normalized_tokens = F.normalize(patches, dim=-1)
    normalized_anchors = F.normalize(model.anchors_img, dim=-1)
    profiles = normalized_tokens @ normalized_anchors.T
    return F.normalize(profiles, dim=-1)


def infer_square_grid(num_patches: int) -> tuple[int, int]:
    side = int(num_patches ** 0.5)
    if side * side != num_patches:
        raise ValueError(f"num_patches must be a square grid, got {num_patches}.")
    return side, side


def dense_patch_logits(
    patch_profiles: torch.Tensor,
    class_profiles: torch.Tensor,
    grid_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Return dense class logits shaped ``(B, C, H_patch, W_patch)``."""

    if patch_profiles.dim() != 3:
        raise ValueError("patch_profiles must be shaped (B, P, K).")
    if class_profiles.dim() != 2:
        raise ValueError("class_profiles must be shaped (C, K).")
    if patch_profiles.shape[-1] != class_profiles.shape[-1]:
        raise ValueError("patch and class profiles must share the K dimension.")
    if grid_size is None:
        grid_size = infer_square_grid(int(patch_profiles.shape[1]))
    h, w = grid_size
    if h * w != patch_profiles.shape[1]:
        raise ValueError("grid_size does not match number of patch profiles.")
    logits = F.normalize(patch_profiles, dim=-1) @ F.normalize(class_profiles, dim=-1).T
    return logits.transpose(1, 2).reshape(patch_profiles.shape[0], class_profiles.shape[0], h, w)


def patch_logits_to_label_mask(
    logits: torch.Tensor,
    output_size: tuple[int, int],
    label_offset: int = 1,
) -> torch.Tensor:
    """Upsample dense patch logits and convert argmax to dataset label ids."""

    if logits.dim() != 4:
        raise ValueError("logits must be shaped (B, C, H, W).")
    upsampled = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
    return upsampled.argmax(dim=1).to(torch.long) + int(label_offset)


def update_intersections_unions(
    intersections: torch.Tensor,
    unions: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_ids: Iterable[int],
    ignore_index: int | None = None,
) -> None:
    """Accumulate per-class intersection/union for segmentation masks."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if intersections.shape != unions.shape:
        raise ValueError("intersections and unions must have the same shape.")
    pred = prediction.detach().cpu()
    tgt = target.detach().cpu()
    valid = torch.ones_like(tgt, dtype=torch.bool)
    if ignore_index is not None:
        valid = tgt != int(ignore_index)
    class_list = [int(item) for item in class_ids]
    if len(class_list) != intersections.numel():
        raise ValueError("class_ids length must match accumulator size.")
    for idx, class_id in enumerate(class_list):
        pred_c = (pred == class_id) & valid
        tgt_c = (tgt == class_id) & valid
        intersections[idx] += (pred_c & tgt_c).sum().item()
        unions[idx] += (pred_c | tgt_c).sum().item()


def foreground_miou_from_intersections_unions(
    intersections: torch.Tensor,
    unions: torch.Tensor,
) -> float:
    """Return foreground mIoU percentage from accumulated intersections/unions."""

    valid = unions > 0
    if not valid.any():
        return 0.0
    return float((intersections[valid] / unions[valid]).mean().item() * 100.0)
