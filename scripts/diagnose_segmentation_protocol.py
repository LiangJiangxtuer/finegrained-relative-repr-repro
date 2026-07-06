#!/usr/bin/env python3
"""Probe dense segmentation protocol assumptions without launching full evals.

The full segmentation runner intentionally mirrors the initial reproduction path. This
script is a lightweight diagnostic loop for root-cause analysis: it compares mIoU
against the original target mask and against a target mask transformed with the same
DINOv2 image-processor resize + center-crop geometry used for patch extraction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from pal_repro.segmentation import (
    dense_patch_logits,
    foreground_miou_from_intersections_unions,
    image_patch_profiles,
    patch_logits_to_label_mask,
    segmentation_prompts,
    transform_mask_like_image_processor,
    update_intersections_unions,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_segmentation import DEFAULT_ROOTS, collate_pil_masks, load_dataset
from pal_repro.evaluate import load_trained_pal_model


def _top_counts(values: torch.Tensor, k: int = 10) -> list[dict[str, int]]:
    unique, counts = torch.unique(values.detach().cpu(), return_counts=True)
    pairs = sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True)
    return [{"label": int(label), "count": int(count)} for label, count in pairs[:k]]


@torch.no_grad()
def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    root = args.root or DEFAULT_ROOTS[args.dataset]
    dataset, class_names, class_ids, default_ignore_index, split = load_dataset(
        args.dataset,
        root,
        context_protocol=args.context_protocol,
    )
    sample_count = min(args.limit, len(dataset))
    dataset = torch.utils.data.Subset(dataset, list(range(sample_count)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pil_masks)

    pal_model = load_trained_pal_model(args.checkpoint, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=args.local_files_only)
    text_model = AutoModel.from_pretrained(args.text_model, local_files_only=args.local_files_only).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model, local_files_only=args.local_files_only)
    vision_model = AutoModel.from_pretrained(args.vision_model, local_files_only=args.local_files_only).to(device).eval()

    prompts = segmentation_prompts(class_names, template=args.prompt_template)
    text_inputs = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_text_length).to(device)
    text_outputs = text_model(**text_inputs)
    class_profiles = pal_model.encode_text(text_outputs.last_hidden_state.float(), text_inputs["attention_mask"].bool()).detach()

    ignore_candidates: list[int | None] = [default_ignore_index]
    if args.also_ignore_zero and 0 not in ignore_candidates:
        ignore_candidates.append(0)
    if None not in ignore_candidates:
        ignore_candidates.insert(0, None)

    accumulators: dict[str, dict[int | None, tuple[torch.Tensor, torch.Tensor]]] = {}
    for frame in ("original_target", "processor_aligned_target"):
        accumulators[frame] = {
            ignore: (torch.zeros(len(class_ids), dtype=torch.float64), torch.zeros(len(class_ids), dtype=torch.float64))
            for ignore in ignore_candidates
        }

    target_top: list[dict[str, int]] = []
    aligned_target_top: list[dict[str, int]] = []
    pred_top: list[dict[str, int]] = []
    processed = 0

    for images, masks in loader:
        image_inputs = image_processor(images=images, return_tensors="pt").to(device)
        image_outputs = vision_model(**image_inputs)
        patch_profiles = image_patch_profiles(pal_model, image_outputs.last_hidden_state.float())
        logits = dense_patch_logits(patch_profiles, class_profiles)
        for index, mask in enumerate(masks):
            target = torch.as_tensor(np.array(mask), dtype=torch.long)
            aligned_mask = transform_mask_like_image_processor(mask, image_processor)
            aligned_target = torch.as_tensor(np.array(aligned_mask), dtype=torch.long)
            pred_original = patch_logits_to_label_mask(
                logits[index : index + 1].detach().cpu(),
                output_size=tuple(target.shape),
                label_ids=class_ids,
            )[0]
            pred_aligned = patch_logits_to_label_mask(
                logits[index : index + 1].detach().cpu(),
                output_size=tuple(aligned_target.shape),
                label_ids=class_ids,
            )[0]

            target_top.extend(_top_counts(target, k=5))
            aligned_target_top.extend(_top_counts(aligned_target, k=5))
            pred_top.extend(_top_counts(pred_aligned, k=5))
            for ignore in ignore_candidates:
                inter, union = accumulators["original_target"][ignore]
                update_intersections_unions(inter, union, pred_original, target, class_ids=class_ids, ignore_index=ignore)
                inter, union = accumulators["processor_aligned_target"][ignore]
                update_intersections_unions(inter, union, pred_aligned, aligned_target, class_ids=class_ids, ignore_index=ignore)
        processed += len(images)
        print(f"processed {processed}/{sample_count}", flush=True)

    def summarize_counter(rows: list[dict[str, int]]) -> list[dict[str, int]]:
        counts: dict[int, int] = {}
        for row in rows:
            counts[row["label"]] = counts.get(row["label"], 0) + row["count"]
        return [{"label": label, "count": count} for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:15]]

    metrics: dict[str, dict[str, float]] = {}
    for frame, by_ignore in accumulators.items():
        metrics[frame] = {}
        for ignore, (inter, union) in by_ignore.items():
            key = "none" if ignore is None else str(ignore)
            metrics[frame][f"ignore_{key}"] = foreground_miou_from_intersections_unions(inter, union)

    return {
        "dataset": args.dataset,
        "split": split,
        "num_samples": sample_count,
        "num_classes": len(class_ids),
        "context_protocol": args.context_protocol if args.dataset == "Context" else None,
        "class_id_min": min(class_ids),
        "class_id_max": max(class_ids),
        "default_ignore_index": default_ignore_index,
        "processor": {
            "do_resize": getattr(image_processor, "do_resize", None),
            "size": getattr(image_processor, "size", None),
            "do_center_crop": getattr(image_processor, "do_center_crop", None),
            "crop_size": getattr(image_processor, "crop_size", None),
        },
        "metrics": metrics,
        "top_target_labels_original": summarize_counter(target_top),
        "top_target_labels_processor_aligned": summarize_counter(aligned_target_top),
        "top_predicted_labels_processor_aligned": summarize_counter(pred_top),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["VOC20", "Context", "ADE20K"], required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-model", default="facebook/dinov2-large")
    parser.add_argument("--text-model", default="roberta-large")
    parser.add_argument("--prompt-template", default="a photo of {class_name}")
    parser.add_argument("--context-protocol", choices=["all459", "common59"], default="all459")
    parser.add_argument("--max-text-length", type=int, default=32)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--also-ignore-zero", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
