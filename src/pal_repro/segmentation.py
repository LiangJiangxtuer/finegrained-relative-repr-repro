"""Dense segmentation helpers for PAL paper reproduction."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

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

PASCAL_CONTEXT_59_CLASS_NAMES: tuple[str, ...] = (
    "aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird",
    "boat", "book", "bottle", "building", "bus", "cabinet", "car", "cat",
    "ceiling", "chair", "cloth", "computer", "cow", "cup", "curtain", "dog",
    "door", "fence", "floor", "flower", "food", "grass", "ground", "horse",
    "keyboard", "light", "motorbike", "mountain", "mouse", "person", "plate",
    "platform", "pottedplant", "road", "rock", "sheep", "shelves", "sidewalk",
    "sign", "sky", "snow", "sofa", "table", "track", "train", "tree", "truck",
    "tvmonitor", "wall", "water", "window", "wood",
)

SEGMENTATION_CLASS_NAME_ALIASES: dict[str, str] = {
    "bedclothes": "bed clothes",
    "pottedplant": "potted plant",
    "tvmonitor": "tv monitor",
}

DEFAULT_SEGMENTATION_PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of {class_name}",
    "a close-up photo of {class_name}",
    "a cropped photo of {class_name}",
    "a clean photo of {class_name}",
)

ADE20K_CLEAN_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    # The ADE20K metadata is WordNet-derived and often appends broad or
    # colloquial synsets (e.g. "person, mortal, soul" or "car, machine").
    # Keep only aliases that are visually interchangeable class names.
    "building": ("building", "edifice"),
    "floor": ("floor", "flooring"),
    "road": ("road",),
    "windowpane": ("windowpane", "window"),
    "sidewalk": ("sidewalk", "pavement"),
    "person": ("person",),
    "earth": ("earth", "ground"),
    "door": ("door", "double door"),
    "mountain": ("mountain",),
    "plant": ("plant",),
    "curtain": ("curtain", "drape", "drapery"),
    "car": ("car", "automobile", "motorcar"),
    "painting": ("painting", "picture"),
    "sofa": ("sofa", "couch"),
    "rug": ("rug", "carpet", "carpeting"),
    "fence": ("fence", "fencing"),
    "rock": ("rock", "stone"),
    "wardrobe": ("wardrobe", "closet"),
    "bathtub": ("bathtub", "bathing tub", "bath", "tub"),
    "railing": ("railing", "rail"),
    "base": ("base", "pedestal"),
    "column": ("column", "pillar"),
    "signboard": ("signboard", "sign"),
    "chest of drawers": ("chest of drawers", "dresser"),
    "fireplace": ("fireplace", "hearth", "open fireplace"),
    "refrigerator": ("refrigerator", "icebox"),
    "grandstand": ("grandstand", "covered stand"),
    "stairs": ("stairs", "steps"),
    "case": ("case", "display case", "showcase", "vitrine"),
    "pool table": ("pool table", "billiard table", "snooker table"),
    "screen door": ("screen door",),
    "stairway": ("stairway", "staircase"),
    "bridge": ("bridge",),
    "blind": ("blind",),
    "coffee table": ("coffee table", "cocktail table"),
    "toilet": ("toilet", "commode"),
    "stove": ("stove", "kitchen stove", "cooking stove"),
    "palm": ("palm", "palm tree"),
    "computer": ("computer", "electronic computer"),
    "hovel": ("hovel", "hut", "shack", "shanty"),
    "bus": ("bus", "motorbus"),
    "light": ("light", "light source"),
    "truck": ("truck", "motortruck"),
    "chandelier": ("chandelier", "pendant"),
    "awning": ("awning", "sunshade", "sunblind"),
    "streetlight": ("streetlight", "street lamp"),
    "booth": ("booth", "kiosk"),
    "television receiver": ("television", "television set", "tv", "tv set"),
    "airplane": ("airplane", "aeroplane", "plane"),
    "apparel": ("apparel", "clothes"),
    "land": ("land", "soil"),
    "bannister": ("bannister", "banister", "handrail"),
    "escalator": ("escalator", "moving staircase", "moving stairway"),
    "ottoman": ("ottoman", "hassock"),
    "buffet": ("buffet", "sideboard"),
    "poster": ("poster", "placard"),
    "conveyer belt": ("conveyer belt", "conveyor belt"),
    "washer": ("washer", "automatic washer", "washing machine"),
    "plaything": ("plaything", "toy"),
    "swimming pool": ("swimming pool",),
    "barrel": ("barrel", "cask"),
    "basket": ("basket", "handbasket"),
    "waterfall": ("waterfall", "falls"),
    "tent": ("tent",),
    "minibike": ("minibike", "motorbike"),
    "food": ("food",),
    "step": ("step", "stair"),
    "tank": ("tank", "storage tank"),
    "trade name": ("trade name", "brand name", "brand"),
    "microwave": ("microwave", "microwave oven"),
    "pot": ("pot", "flowerpot"),
    "animal": ("animal",),
    "bicycle": ("bicycle", "bike"),
    "dishwasher": ("dishwasher", "dish washer", "dishwashing machine"),
    "screen": ("screen", "projection screen"),
    "blanket": ("blanket",),
    "hood": ("hood", "exhaust hood"),
    "traffic light": ("traffic light", "traffic signal", "stoplight"),
    "ashcan": ("ashcan", "trash can", "garbage can", "wastebin", "dustbin", "trash bin"),
    "pier": ("pier", "dock"),
    "monitor": ("monitor",),
    "bulletin board": ("bulletin board", "notice board"),
    "glass": ("glass", "drinking glass"),
}


def normalize_segmentation_class_name(name: str) -> str:
    """Return prompt-friendly class names for dense zero-shot segmentation."""

    stripped = name.strip()
    return SEGMENTATION_CLASS_NAME_ALIASES.get(stripped, stripped.replace("_", " "))


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


def select_pascal_context_labels(
    labels: Iterable[tuple[int, str]],
    protocol: str = "all459",
) -> list[tuple[int, str]]:
    """Select Pascal Context labels for either all-459 or common-59 protocol."""

    rows = list(labels)
    if protocol == "all459":
        return rows
    if protocol != "common59":
        raise ValueError(f"Unsupported Pascal Context protocol: {protocol}")
    by_name = {name: (label_id, name) for label_id, name in rows}
    missing = [name for name in PASCAL_CONTEXT_59_CLASS_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"Missing Pascal Context common59 labels: {missing}")
    return [by_name[name] for name in PASCAL_CONTEXT_59_CLASS_NAMES]


def parse_ade20k_object_info(lines: Iterable[str]) -> list[tuple[int, str]]:
    """Parse ADE20K ``objectInfo150.txt`` rows using the first class-name alias."""

    return [(class_id, aliases[0]) for class_id, aliases in parse_ade20k_object_aliases(lines)]


def parse_ade20k_object_aliases(lines: Iterable[str]) -> list[tuple[int, list[str]]]:
    """Parse ADE20K rows while preserving all comma-separated class aliases."""

    rows: list[tuple[int, list[str]]] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("Idx"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        class_id = int(parts[0].strip())
        aliases = [
            normalize_segmentation_class_name(item)
            for item in parts[-1].split(",")
            if item.strip()
        ]
        if aliases:
            rows.append((class_id, aliases))
    return sorted(rows, key=lambda item: item[0])


def clean_ade20k_object_aliases(
    class_aliases: Iterable[tuple[int, list[str]]],
) -> list[tuple[int, list[str]]]:
    """Return conservative ADE20K prompt aliases, avoiding broad WordNet synonyms."""

    cleaned: list[tuple[int, list[str]]] = []
    for class_id, aliases in class_aliases:
        if not aliases:
            continue
        canonical = aliases[0]
        has_override = canonical in ADE20K_CLEAN_ALIAS_OVERRIDES
        selected = list(ADE20K_CLEAN_ALIAS_OVERRIDES.get(canonical, (canonical,)))
        allowed = set(aliases)
        filtered = [alias for alias in selected if alias in allowed or alias == canonical]
        if not has_override and canonical not in filtered:
            filtered.insert(0, canonical)
        cleaned.append((class_id, list(dict.fromkeys(filtered))))
    return cleaned


class PascalContextSegmentationDataset:
    """Pascal Context trainval dataset backed by ``LabelMap`` .mat masks."""

    def __init__(self, root: str | Path, image_root: str | Path | None = None, protocol: str = "all459") -> None:
        self.root = Path(root)
        self.protocol = protocol
        label_path = self.root / "labels.txt"
        raw_labels = parse_pascal_context_labels(label_path.read_text(encoding="utf-8").splitlines())
        self.labels = select_pascal_context_labels(raw_labels, protocol=protocol)
        self.class_ids = [item[0] for item in self.labels]
        self.class_names = [normalize_segmentation_class_name(item[1]) for item in self.labels]
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
        self.class_aliases = parse_ade20k_object_aliases(info_path.read_text(encoding="utf-8").splitlines())
        self.labels = [(class_id, aliases[0]) for class_id, aliases in self.class_aliases]
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


def build_segmentation_prompt_groups(
    class_aliases: Iterable[Iterable[str] | str],
    templates: Iterable[str] = DEFAULT_SEGMENTATION_PROMPT_TEMPLATES,
) -> list[list[str]]:
    """Build one prompt ensemble per segmentation class."""

    template_list = list(templates)
    if not template_list:
        raise ValueError("templates must contain at least one prompt template.")
    groups: list[list[str]] = []
    for aliases in class_aliases:
        alias_list = [aliases] if isinstance(aliases, str) else list(aliases)
        prompts: list[str] = []
        for alias in alias_list:
            normalized = normalize_segmentation_class_name(alias)
            prompts.extend(template.format(class_name=normalized) for template in template_list)
        groups.append(prompts)
    return groups


def _get_hw(mapping: dict[str, int] | None, default: int) -> tuple[int, int]:
    if not mapping:
        return default, default
    height = int(mapping.get("height", default))
    width = int(mapping.get("width", default))
    return height, width


def transform_mask_like_image_processor(mask: Image.Image, image_processor: Any) -> Image.Image:
    """Apply image-processor resize/center-crop geometry to a label mask."""

    out = mask
    if getattr(image_processor, "do_resize", False):
        size = getattr(image_processor, "size", {}) or {}
        width, height = out.size
        if "shortest_edge" in size:
            shortest = int(size["shortest_edge"])
            if width <= height:
                new_width = shortest
                new_height = int(round(height * shortest / width))
            else:
                new_height = shortest
                new_width = int(round(width * shortest / height))
        else:
            new_height, new_width = _get_hw(size, default=224)
        out = out.resize((new_width, new_height), resample=Image.Resampling.NEAREST)

    if getattr(image_processor, "do_center_crop", False):
        crop_height, crop_width = _get_hw(getattr(image_processor, "crop_size", None), default=224)
        width, height = out.size
        left = max((width - crop_width) // 2, 0)
        top = max((height - crop_height) // 2, 0)
        out = out.crop((left, top, left + crop_width, top + crop_height))
    return out


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


def calibrate_dense_logits(
    logits: torch.Tensor,
    mode: str = "none",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply diagnostic dense-logit calibration before argmax decoding.

    These modes are intended for protocol/debug probes, not as paper-grade
    defaults. They operate per image and per class so they can test whether
    class-specific logit offsets or dynamic ranges dominate dense predictions.
    """

    if mode == "none":
        return logits
    if logits.dim() != 4:
        raise ValueError("logits must be shaped (B, C, H, W).")
    if mode == "image-class-center":
        return logits - logits.mean(dim=(2, 3), keepdim=True)
    if mode == "image-class-zscore":
        centered = logits - logits.mean(dim=(2, 3), keepdim=True)
        scale = logits.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(float(eps))
        return centered / scale
    raise ValueError(f"Unsupported dense logit calibration mode: {mode}")


def patch_logits_to_label_mask(
    logits: torch.Tensor,
    output_size: tuple[int, int],
    label_offset: int = 1,
    label_ids: Iterable[int] | None = None,
) -> torch.Tensor:
    """Upsample dense patch logits and convert argmax to dataset label ids."""

    if logits.dim() != 4:
        raise ValueError("logits must be shaped (B, C, H, W).")
    upsampled = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
    argmax = upsampled.argmax(dim=1).to(torch.long)
    if label_ids is None:
        return argmax + int(label_offset)
    labels = torch.as_tensor([int(item) for item in label_ids], dtype=torch.long, device=argmax.device)
    if labels.numel() != logits.shape[1]:
        raise ValueError("label_ids length must match the number of logit classes.")
    return labels[argmax]


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
