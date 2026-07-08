#!/usr/bin/env python3
"""Evaluate PAL dense zero-shot segmentation.

Currently supports the VOC20 foreground mIoU experiment. The runner uses DINOv2
patch tokens, maps each patch to PAL image-anchor similarities, compares patch
profiles to PAL text profiles for class prompts, upsamples patch logits to mask
resolution, and reports foreground mIoU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np

from pal_repro.evaluate import load_trained_pal_model
from pal_repro.segmentation import (
    ADE20KSegmentationDataset,
    PascalContextSegmentationDataset,
    VOC_CLASS_NAMES,
    build_segmentation_prompt_groups,
    calibrate_dense_logits,
    clean_ade20k_object_aliases,
    dense_patch_logits,
    foreground_miou_from_intersections_unions,
    image_patch_profiles,
    patch_logits_to_label_mask,
    segmentation_prompts,
    transform_mask_like_image_processor,
    update_intersections_unions,
)

DEFAULT_ROOTS = {
    "VOC20": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/segmentation/voc2012"),
    "Context": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/segmentation/pascal_context/raw"),
    "ADE20K": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/segmentation/ade20k/raw/ADEChallengeData2016"),
}


def collate_pil_masks(batch):
    images, masks = zip(*batch)
    return list(images), list(masks)


def load_dataset(name: str, root: Path, context_protocol: str = "all459"):
    if name == "VOC20":
        ds = datasets.VOCSegmentation(root=str(root), year="2012", image_set="val", download=False)
        return ds, VOC_CLASS_NAMES, list(range(1, 21)), 255, "val"
    if name == "Context":
        ds = PascalContextSegmentationDataset(root, protocol=context_protocol)
        return ds, ds.class_names, ds.class_ids, None, "trainval"
    if name == "ADE20K":
        ds = ADE20KSegmentationDataset(root)
        return ds, ds.class_names, ds.class_ids, None, "validation"
    raise ValueError(f"Unsupported dataset: {name}")


@torch.no_grad()
def encode_class_profiles(
    pal_model,
    tokenizer,
    text_model,
    prompts: list[str],
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = text_model(**inputs)
    return pal_model.encode_text(outputs.last_hidden_state.float(), inputs["attention_mask"].bool()).detach().cpu()


def select_hidden_state(outputs: Any, layer: int | None) -> torch.Tensor:
    """Return last_hidden_state or a requested hidden-state layer."""

    if layer is None:
        return outputs.last_hidden_state
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model output did not include hidden_states; pass output_hidden_states=True.")
    return hidden_states[layer]


def select_token_features(
    outputs: Any,
    layer: int | None,
    layer_ensemble: list[int] | None = None,
) -> torch.Tensor:
    """Return a single token tensor, optionally averaging hidden-state layers."""

    if layer_ensemble:
        states = [select_hidden_state(outputs, item).float() for item in layer_ensemble]
        return torch.stack(states, dim=0).mean(dim=0)
    return select_hidden_state(outputs, layer).float()


def override_image_processor_size(image_processor: Any, image_size: int | None) -> None:
    """Optionally evaluate dense segmentation at a larger square processor crop."""

    if image_size is None:
        return
    size = int(image_size)
    image_processor.size = {"shortest_edge": size}
    image_processor.crop_size = {"height": size, "width": size}


def class_aliases_for_dataset(source_dataset: Any, class_names: list[str], alias_policy: str) -> list[list[str]]:
    if alias_policy == "all" and hasattr(source_dataset, "class_aliases"):
        return [aliases for _class_id, aliases in source_dataset.class_aliases]
    if alias_policy == "clean" and hasattr(source_dataset, "class_aliases"):
        return [aliases for _class_id, aliases in clean_ade20k_object_aliases(source_dataset.class_aliases)]
    return [[name] for name in class_names]


def ade20k_prior_bias(
    dataset_name: str,
    root: Path,
    class_ids: list[int],
    source: str,
    alpha: float,
    device: torch.device,
) -> torch.Tensor | None:
    if source == "none" or alpha == 0.0:
        return None
    if dataset_name != "ADE20K":
        raise ValueError("class-prior correction is currently implemented only for ADE20K")
    values: dict[int, float] = {}
    for raw in (root / "objectInfo150.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("Idx"):
            continue
        parts = line.split("\t")
        class_id = int(parts[0])
        if source == "ade20k-ratio":
            values[class_id] = float(parts[1])
        elif source == "ade20k-train-count":
            values[class_id] = float(parts[2])
        else:
            raise ValueError(f"Unsupported class prior source: {source}")
    prior = torch.tensor([values[int(class_id)] for class_id in class_ids], dtype=torch.float32, device=device)
    prior = prior / prior.sum().clamp_min(1e-12)
    bias = torch.log(prior.clamp_min(1e-12))
    bias = bias - bias.mean()
    return (float(alpha) * bias).view(1, -1, 1, 1)


def manual_class_bias(
    class_names: list[str],
    specs: list[str] | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Build a dense-logit bias tensor from explicit class/group specs.

    Spec format is ``class name=value`` or ``class a,class b=value``. This is
    intentionally explicit so diagnostic calibration probes can target a small
    class group without introducing a global dataset prior.
    """

    if not specs:
        return None
    name_to_index = {name: idx for idx, name in enumerate(class_names)}
    bias = torch.zeros(len(class_names), dtype=torch.float32, device=device)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Class-bias spec must be NAME=VALUE, got: {spec!r}")
        names_text, value_text = spec.split("=", 1)
        value = float(value_text)
        names = [item.strip() for item in names_text.split(",") if item.strip()]
        if not names:
            raise ValueError(f"Class-bias spec has no class names: {spec!r}")
        for name in names:
            if name not in name_to_index:
                raise KeyError(f"Unknown class name in --class-bias: {name!r}")
            bias[name_to_index[name]] += value
    return bias.view(1, -1, 1, 1)


@torch.no_grad()
def encode_class_profile_groups(
    pal_model,
    tokenizer,
    text_model,
    prompt_groups: list[list[str]],
    device: torch.device,
    max_length: int,
    text_layer: int | None,
    text_layer_ensemble: list[int] | None = None,
) -> torch.Tensor:
    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    inputs = tokenizer(
        flat_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = text_model(
        **inputs,
        output_hidden_states=text_layer is not None or bool(text_layer_ensemble),
    )
    text_tokens = select_token_features(outputs, text_layer, text_layer_ensemble)
    flat_features = torch.nn.functional.normalize(
        pal_model.encode_text(text_tokens, inputs["attention_mask"].bool()).detach().cpu(),
        dim=-1,
    )
    grouped: list[torch.Tensor] = []
    offset = 0
    for group in prompt_groups:
        width = len(group)
        grouped.append(torch.nn.functional.normalize(flat_features[offset:offset + width].mean(dim=0), dim=0))
        offset += width
    return torch.stack(grouped, dim=0)


@torch.no_grad()
def evaluate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    root = args.root or DEFAULT_ROOTS[args.dataset]
    dataset, class_names, class_ids, ignore_index, split = load_dataset(
        args.dataset,
        root,
        context_protocol=args.context_protocol,
    )
    if args.ignore_zero:
        ignore_index = 0
    if args.limit is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil_masks,
    )

    pal_model = load_trained_pal_model(args.checkpoint, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=args.local_files_only)
    text_model = AutoModel.from_pretrained(args.text_model, local_files_only=args.local_files_only).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model, local_files_only=args.local_files_only)
    override_image_processor_size(image_processor, args.image_size)
    vision_model = AutoModel.from_pretrained(args.vision_model, local_files_only=args.local_files_only).to(device).eval()

    templates = args.prompt_template or ["a photo of {class_name}"]
    source_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    class_aliases = class_aliases_for_dataset(source_dataset, class_names, args.alias_policy)
    prompt_groups = build_segmentation_prompt_groups(class_aliases, templates=templates)
    prompts = [prompt for group in prompt_groups for prompt in group]
    class_profiles = encode_class_profile_groups(
        pal_model,
        tokenizer,
        text_model,
        prompt_groups,
        device=device,
        max_length=args.max_text_length,
        text_layer=args.text_layer,
        text_layer_ensemble=args.text_layer_ensemble,
    ).to(device)
    prior_bias = ade20k_prior_bias(
        args.dataset,
        root,
        class_ids,
        source=args.class_prior_source,
        alpha=args.class_prior_alpha,
        device=device,
    )
    class_bias = manual_class_bias(class_names, args.class_bias, device=device)

    intersections = torch.zeros(len(class_ids), dtype=torch.float64)
    unions = torch.zeros(len(class_ids), dtype=torch.float64)
    pred_counts = torch.zeros(len(class_ids), dtype=torch.float64)
    target_counts = torch.zeros(len(class_ids), dtype=torch.float64)
    confusion_samples: list[dict[str, Any]] = []
    processed = 0
    for images, masks in loader:
        image_inputs = image_processor(images=images, return_tensors="pt").to(device)
        image_outputs = vision_model(
            **image_inputs,
            output_hidden_states=args.vision_layer is not None or bool(args.vision_layer_ensemble),
        )
        image_tokens = select_token_features(image_outputs, args.vision_layer, args.vision_layer_ensemble)
        patch_profiles = image_patch_profiles(pal_model, image_tokens)
        logits = dense_patch_logits(patch_profiles, class_profiles)
        if prior_bias is not None:
            logits = logits + prior_bias
        if class_bias is not None:
            logits = logits + class_bias
        logits = calibrate_dense_logits(logits, mode=args.logit_calibration)
        for index, mask in enumerate(masks):
            if args.target_frame == "processor":
                mask = transform_mask_like_image_processor(mask, image_processor)
            target = torch.as_tensor(np.array(mask), dtype=torch.long)
            pred = patch_logits_to_label_mask(
                logits[index:index + 1].detach().cpu(),
                output_size=tuple(target.shape),
                label_ids=class_ids,
            )[0]
            update_intersections_unions(
                intersections,
                unions,
                pred,
                target,
                class_ids=class_ids,
                ignore_index=ignore_index,
            )
            valid_target = target.flatten()
            valid_mask = torch.ones_like(target, dtype=torch.bool)
            if ignore_index is not None:
                valid_mask = target != int(ignore_index)
                valid_target = target[valid_mask].flatten()
            for class_index, class_id in enumerate(class_ids):
                pred_counts[class_index] += ((pred == int(class_id)) & valid_mask).sum().item()
                target_counts[class_index] += (valid_target == int(class_id)).sum().item()
            if len(confusion_samples) < 8:
                pred_top = int(torch.bincount(pred.flatten()).argmax().item())
                target_top = int(torch.bincount(valid_target).argmax().item()) if valid_target.numel() else -1
                confusion_samples.append({"pred_top_label": pred_top, "target_top_label": target_top})
        processed += len(images)
        print(f"processed {processed}/{len(dataset)}", flush=True)

    class_ious = {}
    for idx, class_name in enumerate(class_names):
        if unions[idx] > 0:
            class_ious[class_name] = float((intersections[idx] / unions[idx]).item() * 100.0)
        else:
            class_ious[class_name] = None
    metrics = {
        "foreground_miou": foreground_miou_from_intersections_unions(intersections, unions),
    }
    top_predicted_classes = [
        {"class_name": class_names[idx], "pixels": float(pred_counts[idx].item())}
        for idx in torch.argsort(pred_counts, descending=True)[:10].tolist()
    ]
    target_frequency = {
        class_names[idx]: float(target_counts[idx].item())
        for idx in torch.argsort(target_counts, descending=True)[:10].tolist()
    }
    class_counts = {
        class_names[idx]: {
            "intersection": float(intersections[idx].item()),
            "union": float(unions[idx].item()),
            "pred_pixels": float(pred_counts[idx].item()),
            "target_pixels": float(target_counts[idx].item()),
            "iou": class_ious[class_names[idx]],
        }
        for idx in range(len(class_names))
    }
    result: dict[str, Any] = {
        "dataset": args.dataset,
        "split": split,
        "root": str(root),
        "checkpoint": str(args.checkpoint),
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "vision_layer": args.vision_layer,
        "text_layer": args.text_layer,
        "vision_layer_ensemble": args.vision_layer_ensemble,
        "text_layer_ensemble": args.text_layer_ensemble,
        "num_samples": len(dataset),
        "num_classes": len(class_names),
        "context_protocol": args.context_protocol if args.dataset == "Context" else None,
        "target_frame": args.target_frame,
        "alias_policy": args.alias_policy,
        "class_prior_source": args.class_prior_source,
        "class_prior_alpha": args.class_prior_alpha,
        "class_bias": args.class_bias,
        "logit_calibration": args.logit_calibration,
        "prompt_template": templates[0] if len(templates) == 1 else None,
        "prompt_templates": templates,
        "prompt_groups": prompt_groups,
        "prompts": prompts,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "device": str(device),
        "ignore_index": ignore_index,
        "metrics": metrics,
        "class_ious": class_ious,
        "class_counts": class_counts,
        "top_predicted_classes": top_predicted_classes,
        "target_frequency": target_frequency,
        "confusion_samples": confusion_samples,
        "intersections": [float(item) for item in intersections.tolist()],
        "unions": [float(item) for item in unions.tolist()],
        "protocol": {
            "source_paper_claim": "Table 3 zero-shot segmentation mIoU-fg",
            "encoder_layers": {
                "vision_layer": args.vision_layer,
                "text_layer": args.text_layer,
                "vision_layer_ensemble": args.vision_layer_ensemble,
                "text_layer_ensemble": args.text_layer_ensemble,
            },
            "dataset_split": split,
            "target_frame": args.target_frame,
            "image_size": args.image_size,
            "context_protocol": args.context_protocol if args.dataset == "Context" else None,
            "prompt_policy": {
                "templates": templates,
                "alias_policy": args.alias_policy,
                "aggregation": "normalized_mean_per_class",
            },
            "class_prior": {"source": args.class_prior_source, "alpha": args.class_prior_alpha},
            "class_bias": args.class_bias,
            "logit_calibration": args.logit_calibration,
            "known_deviation": [],
            "verification_status": "ANALYZED",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_ROOTS), required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-model", default="facebook/dinov2-large")
    parser.add_argument("--text-model", default="roberta-large")
    parser.add_argument("--vision-layer", type=int, default=None)
    parser.add_argument("--text-layer", type=int, default=None)
    parser.add_argument("--vision-layer-ensemble", action="append", type=int, default=None, help="Average one or more requested vision hidden-state layers for dense-token protocol probes.")
    parser.add_argument("--text-layer-ensemble", action="append", type=int, default=None, help="Average one or more requested text hidden-state layers for dense-token protocol probes.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=None, help="Override DINOv2 processor shortest-edge and square crop size for dense evaluation.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=32)
    parser.add_argument("--prompt-template", action="append", default=None)
    parser.add_argument("--alias-policy", choices=["first", "all", "clean", "manual"], default="first")
    parser.add_argument("--class-prior-source", choices=["none", "ade20k-ratio", "ade20k-train-count"], default="none")
    parser.add_argument("--class-prior-alpha", type=float, default=0.0)
    parser.add_argument("--class-bias", action="append", default=None, help="Explicit diagnostic class/group logit bias, e.g. 'wall,sky=0.02' or 'screen door=-0.03'.")
    parser.add_argument("--logit-calibration", choices=["none", "image-class-center", "image-class-zscore"], default="none")
    parser.add_argument("--target-frame", choices=["original", "processor"], default="original")
    parser.add_argument("--context-protocol", choices=["all459", "common59"], default="all459")
    parser.add_argument("--ignore-zero", action="store_true", help="Treat label id 0 as void/ignore during mIoU accumulation.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate_dataset(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
