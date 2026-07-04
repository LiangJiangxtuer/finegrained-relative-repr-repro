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
    dense_patch_logits,
    foreground_miou_from_intersections_unions,
    image_patch_profiles,
    patch_logits_to_label_mask,
    segmentation_prompts,
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


def load_dataset(name: str, root: Path):
    if name == "VOC20":
        ds = datasets.VOCSegmentation(root=str(root), year="2012", image_set="val", download=False)
        return ds, VOC_CLASS_NAMES, list(range(1, 21)), 255, "val"
    if name == "Context":
        ds = PascalContextSegmentationDataset(root)
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


@torch.no_grad()
def evaluate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    root = args.root or DEFAULT_ROOTS[args.dataset]
    dataset, class_names, class_ids, ignore_index, split = load_dataset(args.dataset, root)
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

    prompts = segmentation_prompts(class_names, template=args.prompt_template)
    class_profiles = encode_class_profiles(
        pal_model,
        tokenizer,
        text_model,
        prompts,
        device=device,
        max_length=args.max_text_length,
    ).to(device)

    intersections = torch.zeros(len(class_ids), dtype=torch.float64)
    unions = torch.zeros(len(class_ids), dtype=torch.float64)
    processed = 0
    for images, masks in loader:
        image_inputs = image_processor(images=images, return_tensors="pt").to(device)
        image_outputs = vision_model(**image_inputs)
        patch_profiles = image_patch_profiles(pal_model, image_outputs.last_hidden_state.float())
        logits = dense_patch_logits(patch_profiles, class_profiles)
        for index, mask in enumerate(masks):
            target = torch.as_tensor(np.array(mask), dtype=torch.long)
            pred = patch_logits_to_label_mask(
                logits[index:index + 1].detach().cpu(),
                output_size=tuple(target.shape),
                label_offset=1,
            )[0]
            update_intersections_unions(
                intersections,
                unions,
                pred,
                target,
                class_ids=class_ids,
                ignore_index=ignore_index,
            )
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
    result: dict[str, Any] = {
        "dataset": args.dataset,
        "split": split,
        "root": str(root),
        "checkpoint": str(args.checkpoint),
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "num_samples": len(dataset),
        "num_classes": len(class_names),
        "prompt_template": args.prompt_template,
        "prompts": prompts,
        "batch_size": args.batch_size,
        "device": str(device),
        "ignore_index": ignore_index,
        "metrics": metrics,
        "class_ious": class_ious,
        "intersections": [float(item) for item in intersections.tolist()],
        "unions": [float(item) for item in unions.tolist()],
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=32)
    parser.add_argument("--prompt-template", default="a photo of {class_name}")
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
