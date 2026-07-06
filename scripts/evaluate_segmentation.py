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


@torch.no_grad()
def encode_class_profile_groups(
    pal_model,
    tokenizer,
    text_model,
    prompt_groups: list[list[str]],
    device: torch.device,
    max_length: int,
    text_layer: int | None,
) -> torch.Tensor:
    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    inputs = tokenizer(
        flat_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = text_model(**inputs, output_hidden_states=text_layer is not None)
    text_tokens = select_hidden_state(outputs, text_layer).float()
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
    vision_model = AutoModel.from_pretrained(args.vision_model, local_files_only=args.local_files_only).to(device).eval()

    templates = args.prompt_template or ["a photo of {class_name}"]
    source_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
    if args.alias_policy == "all" and hasattr(source_dataset, "class_aliases"):
        class_aliases = [aliases for _class_id, aliases in source_dataset.class_aliases]
    else:
        class_aliases = [[name] for name in class_names]
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
    ).to(device)

    intersections = torch.zeros(len(class_ids), dtype=torch.float64)
    unions = torch.zeros(len(class_ids), dtype=torch.float64)
    pred_counts = torch.zeros(len(class_ids), dtype=torch.float64)
    target_counts = torch.zeros(len(class_ids), dtype=torch.float64)
    confusion_samples: list[dict[str, Any]] = []
    processed = 0
    for images, masks in loader:
        image_inputs = image_processor(images=images, return_tensors="pt").to(device)
        image_outputs = vision_model(**image_inputs, output_hidden_states=args.vision_layer is not None)
        image_tokens = select_hidden_state(image_outputs, args.vision_layer).float()
        patch_profiles = image_patch_profiles(pal_model, image_tokens)
        logits = dense_patch_logits(patch_profiles, class_profiles)
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
            if ignore_index is not None:
                valid_target = valid_target[valid_target != int(ignore_index)]
            for class_index, class_id in enumerate(class_ids):
                pred_counts[class_index] += (pred == int(class_id)).sum().item()
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
    result: dict[str, Any] = {
        "dataset": args.dataset,
        "split": split,
        "root": str(root),
        "checkpoint": str(args.checkpoint),
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "vision_layer": args.vision_layer,
        "text_layer": args.text_layer,
        "num_samples": len(dataset),
        "num_classes": len(class_names),
        "context_protocol": args.context_protocol if args.dataset == "Context" else None,
        "target_frame": args.target_frame,
        "alias_policy": args.alias_policy,
        "prompt_template": templates[0] if len(templates) == 1 else None,
        "prompt_templates": templates,
        "prompt_groups": prompt_groups,
        "prompts": prompts,
        "batch_size": args.batch_size,
        "device": str(device),
        "ignore_index": ignore_index,
        "metrics": metrics,
        "class_ious": class_ious,
        "top_predicted_classes": top_predicted_classes,
        "target_frequency": target_frequency,
        "confusion_samples": confusion_samples,
        "intersections": [float(item) for item in intersections.tolist()],
        "unions": [float(item) for item in unions.tolist()],
        "protocol": {
            "source_paper_claim": "Table 3 zero-shot segmentation mIoU-fg",
            "encoder_layers": {"vision_layer": args.vision_layer, "text_layer": args.text_layer},
            "dataset_split": split,
            "target_frame": args.target_frame,
            "context_protocol": args.context_protocol if args.dataset == "Context" else None,
            "prompt_policy": {
                "templates": templates,
                "alias_policy": args.alias_policy,
                "aggregation": "normalized_mean_per_class",
            },
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=32)
    parser.add_argument("--prompt-template", action="append", default=None)
    parser.add_argument("--alias-policy", choices=["first", "all", "manual"], default="first")
    parser.add_argument("--target-frame", choices=["original", "processor"], default="original")
    parser.add_argument("--context-protocol", choices=["all459", "common59"], default="all459")
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
