#!/usr/bin/env python3
"""Fast multi-checkpoint segmentation probes for PAL ablations.

This shares frozen DINOv2 image forwards across trained PAL ablation checkpoints
for small corrected-protocol dense segmentation probes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from pal_repro.evaluate import load_trained_pal_model
from pal_repro.segmentation import (
    ADE20KSegmentationDataset,
    PascalContextSegmentationDataset,
    VOC_CLASS_NAMES,
    build_segmentation_prompt_groups,
    clean_ade20k_object_aliases,
    dense_patch_logits,
    foreground_miou_from_intersections_unions,
    image_patch_profiles,
    patch_logits_to_label_mask,
    transform_mask_like_image_processor,
    update_intersections_unions,
)

DEFAULT_ROOTS = {
    "VOC20": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/segmentation/voc2012"),
    "Context": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/segmentation/pascal_context/raw"),
    "ADE20K": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/segmentation/ade20k/raw/ADEChallengeData2016"),
}
DEFAULT_DATASETS = ["VOC20", "Context", "ADE20K"]
TEMPLATES = [
    "a photo of {class_name}",
    "a cropped photo of {class_name}",
    "a close-up photo of {class_name}",
    "a clean photo of {class_name}",
]
PAPER_SEG_TARGETS = {"VOC20": 32.3, "Context": 25.5, "ADE20K": 13.8}


@dataclass(frozen=True)
class Variant:
    group: str
    label: str
    checkpoint: Path

    @property
    def name(self) -> str:
        return f"{self.group}_{self.label}"


@dataclass
class SegMetrics:
    intersections: torch.Tensor
    unions: torch.Tensor
    pred_counts: torch.Tensor
    target_counts: torch.Tensor
    num_samples: int

    @classmethod
    def create(cls, num_classes: int) -> "SegMetrics":
        return cls(
            intersections=torch.zeros(num_classes, dtype=torch.float64),
            unions=torch.zeros(num_classes, dtype=torch.float64),
            pred_counts=torch.zeros(num_classes, dtype=torch.float64),
            target_counts=torch.zeros(num_classes, dtype=torch.float64),
            num_samples=0,
        )


def build_variants(root: Path) -> list[Variant]:
    return [
        *(Variant("k", str(k), root / f"outputs/ablations/k_{k}/checkpoint.pt") for k in (32, 64, 128, 256, 512)),
        Variant("tau", "0_02", root / "outputs/ablations/tau_0_02/checkpoint.pt"),
        Variant("tau", "0_03", root / "outputs/pal_k512_coco2014_full/checkpoint.pt"),
        Variant("tau", "0_05", root / "outputs/ablations/tau_0_05/checkpoint.pt"),
        Variant("tau", "0_07", root / "outputs/ablations/tau_0_07/checkpoint.pt"),
        Variant("tau", "0_10", root / "outputs/ablations/tau_0_10/checkpoint.pt"),
        Variant("token_usage", "global", root / "outputs/ablations/token_usage_global/checkpoint.pt"),
        Variant("token_usage", "mean", root / "outputs/ablations/token_usage_mean/checkpoint.pt"),
        Variant("token_usage", "cap", root / "outputs/ablations/token_usage_cap/checkpoint.pt"),
    ]


def select_hidden_state(outputs: Any, layer: int | None) -> torch.Tensor:
    if layer is None:
        return outputs.last_hidden_state
    if outputs.hidden_states is None:
        raise ValueError("hidden states were not returned")
    return outputs.hidden_states[layer]


def collate_pil_masks(batch):
    images, masks = zip(*batch)
    return list(images), list(masks)


def load_dataset(name: str, root: Path, context_protocol: str):
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


def variant_output_dir(root: Path, output_dir: Path, variant: Variant) -> Path:
    return root / output_dir / variant.group / variant.label


def dataset_output_path(root: Path, output_dir: Path, variant: Variant, dataset: str, limit: int | None) -> Path:
    suffix = f"limit{limit}" if limit is not None else "full"
    return variant_output_dir(root, output_dir, variant) / f"{dataset.lower()}_{suffix}_segmentation.json"


def evaluation_mode(limit: int | None) -> str:
    if limit is None:
        return "shared_image_forward_corrected_full_segmentation"
    return "shared_image_forward_corrected_segmentation_probe"


def known_deviation(limit: int | None) -> str:
    if limit is None:
        return "Selected full corrected-protocol ablation; not all sweep checkpoints were fully rerun."
    return "Small-sample corrected-protocol probe; not a full paper-grade segmentation ablation."


def select_variants(args: argparse.Namespace) -> list[Variant]:
    variants = build_variants(args.root)
    if args.only:
        wanted = set(args.only)
        variants = [variant for variant in variants if variant.name in wanted]
        missing = wanted - {variant.name for variant in variants}
        if missing:
            raise KeyError(f"unknown variants: {sorted(missing)}")
    for variant in variants:
        if not variant.checkpoint.exists():
            raise FileNotFoundError(f"missing checkpoint for {variant.name}: {variant.checkpoint}")
    return variants


@torch.no_grad()
def encode_class_features(
    variants: list[Variant],
    models: dict[str, Any],
    tokenizer,
    text_model,
    prompt_groups: list[list[str]],
    device: torch.device,
    max_length: int,
    text_layer: int | None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    inputs = tokenizer(flat_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
    outputs = text_model(**inputs, output_hidden_states=text_layer is not None)
    text_tokens = select_hidden_state(outputs, text_layer).float()
    text_mask = inputs["attention_mask"].bool()
    features: dict[str, torch.Tensor] = {}
    for variant in variants:
        flat_features = F.normalize(models[variant.name].encode_text(text_tokens, text_mask), dim=-1)
        grouped: list[torch.Tensor] = []
        offset = 0
        for group in prompt_groups:
            width = len(group)
            grouped.append(F.normalize(flat_features[offset:offset + width].mean(dim=0), dim=0))
            offset += width
        features[variant.name] = torch.stack(grouped, dim=0)
    return features, flat_prompts


def class_aliases_for_dataset(dataset: Any, class_names: list[str], alias_policy: str) -> list[list[str]]:
    source = dataset.dataset if hasattr(dataset, "dataset") else dataset
    if alias_policy == "all" and hasattr(source, "class_aliases"):
        return [aliases for _class_id, aliases in source.class_aliases]
    if alias_policy == "clean" and hasattr(source, "class_aliases"):
        return [aliases for _class_id, aliases in clean_ade20k_object_aliases(source.class_aliases)]
    return [[name] for name in class_names]


@torch.no_grad()
def evaluate_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    variants: list[Variant],
    models: dict[str, Any],
    tokenizer,
    text_model,
    image_processor,
    vision_model,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    context_protocol = "common59" if dataset_name == "Context" else "all459"
    base_dataset, class_names, class_ids, ignore_index, split = load_dataset(dataset_name, DEFAULT_ROOTS[dataset_name], context_protocol)
    if args.ignore_zero:
        ignore_index = 0
    dataset = base_dataset
    if args.limit is not None:
        dataset = torch.utils.data.Subset(base_dataset, list(range(min(args.limit, len(base_dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_pil_masks)
    aliases = class_aliases_for_dataset(base_dataset, class_names, args.alias_policy)
    prompt_groups = build_segmentation_prompt_groups(aliases, templates=TEMPLATES)
    class_features, prompts = encode_class_features(
        variants,
        models,
        tokenizer,
        text_model,
        prompt_groups,
        device=device,
        max_length=args.max_text_length,
        text_layer=args.text_layer,
    )
    metrics = {variant.name: SegMetrics.create(len(class_ids)) for variant in variants}
    processed = 0
    for images, masks in loader:
        image_inputs = image_processor(images=images, return_tensors="pt").to(device)
        image_outputs = vision_model(**image_inputs, output_hidden_states=args.vision_layer is not None)
        image_tokens = select_hidden_state(image_outputs, args.vision_layer).float()
        targets: list[torch.Tensor] = []
        for mask in masks:
            target_mask = transform_mask_like_image_processor(mask, image_processor) if args.target_frame == "processor" else mask
            targets.append(torch.as_tensor(np.array(target_mask), dtype=torch.long))
        for variant in variants:
            patch_profiles = image_patch_profiles(models[variant.name], image_tokens)
            logits = dense_patch_logits(patch_profiles, class_features[variant.name])
            for index, target in enumerate(targets):
                pred = patch_logits_to_label_mask(
                    logits[index:index + 1].detach().cpu(),
                    output_size=tuple(target.shape),
                    label_ids=class_ids,
                )[0]
                row = metrics[variant.name]
                update_intersections_unions(row.intersections, row.unions, pred, target, class_ids=class_ids, ignore_index=ignore_index)
                valid_target = target.flatten()
                valid_mask = torch.ones_like(target, dtype=torch.bool)
                if ignore_index is not None:
                    valid_mask = target != int(ignore_index)
                    valid_target = target[valid_mask].flatten()
                for class_index, class_id in enumerate(class_ids):
                    row.pred_counts[class_index] += ((pred == int(class_id)) & valid_mask).sum().item()
                    row.target_counts[class_index] += (valid_target == int(class_id)).sum().item()
                row.num_samples += 1
        processed += len(images)
        print(f"{dataset_name}: processed {processed}/{len(dataset)}", flush=True)
    results: dict[str, dict[str, Any]] = {}
    for variant in variants:
        row = metrics[variant.name]
        class_ious = {
            class_names[idx]: (float((row.intersections[idx] / row.unions[idx]).item() * 100.0) if row.unions[idx] > 0 else None)
            for idx in range(len(class_names))
        }
        result = {
            "dataset": dataset_name,
            "split": split,
            "root": str(DEFAULT_ROOTS[dataset_name]),
            "checkpoint": str(variant.checkpoint),
            "variant": variant.name,
            "group": variant.group,
            "label": variant.label,
            "vision_model": args.vision_model,
            "text_model": args.text_model,
            "vision_layer": args.vision_layer,
            "text_layer": args.text_layer,
            "num_samples": row.num_samples,
            "num_classes": len(class_names),
            "context_protocol": context_protocol if dataset_name == "Context" else None,
            "target_frame": args.target_frame,
            "alias_policy": args.alias_policy,
            "prompt_templates": TEMPLATES,
            "prompt_groups": prompt_groups,
            "prompts": prompts,
            "batch_size": args.batch_size,
            "device": str(device),
            "ignore_zero": args.ignore_zero,
            "ignore_index": ignore_index,
            "metrics": {"foreground_miou": foreground_miou_from_intersections_unions(row.intersections, row.unions)},
            "paper_miou": PAPER_SEG_TARGETS[dataset_name],
            "class_ious": class_ious,
            "class_counts": {
                class_names[idx]: {
                    "intersection": float(row.intersections[idx].item()),
                    "union": float(row.unions[idx].item()),
                    "pred_pixels": float(row.pred_counts[idx].item()),
                    "target_pixels": float(row.target_counts[idx].item()),
                    "iou": class_ious[class_names[idx]],
                }
                for idx in range(len(class_names))
            },
            "top_predicted_classes": [
                {"class_name": class_names[idx], "pixels": float(row.pred_counts[idx].item())}
                for idx in torch.argsort(row.pred_counts, descending=True)[:10].tolist()
            ],
            "target_frequency": {
                class_names[idx]: float(row.target_counts[idx].item())
                for idx in torch.argsort(row.target_counts, descending=True)[:10].tolist()
            },
            "intersections": [float(item) for item in row.intersections.tolist()],
            "unions": [float(item) for item in row.unions.tolist()],
            "protocol": {
                "source_paper_claim": "Table 3 zero-shot segmentation mIoU-fg / ablation avg_seg.",
                "shared_image_forward": True,
                "known_deviation": known_deviation(args.limit),
            },
        }
        output = dataset_output_path(args.root, args.output_dir, variant, dataset_name, args.limit)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        results[variant.name] = result
    return results


def write_summaries(args: argparse.Namespace, variants: list[Variant], all_results: dict[str, dict[str, dict[str, Any]]]) -> None:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = []
        for dataset_name in args.dataset:
            result = all_results[dataset_name][variant.name]
            miou = float(result["metrics"]["foreground_miou"])
            variant_rows.append({
                "dataset": dataset_name,
                "num_samples": result["num_samples"],
                "output": str(dataset_output_path(args.root, args.output_dir, variant, dataset_name, args.limit)),
                "foreground_miou": miou,
                "paper_miou": PAPER_SEG_TARGETS[dataset_name],
                "relative_percent": miou / PAPER_SEG_TARGETS[dataset_name] * 100.0,
            })
        average_miou = sum(row["foreground_miou"] for row in variant_rows) / max(len(variant_rows), 1)
        paper_average = sum(PAPER_SEG_TARGETS[name] for name in args.dataset) / max(len(args.dataset), 1)
        summary = {
            "variant": variant.name,
            "group": variant.group,
            "label": variant.label,
            "checkpoint": str(variant.checkpoint),
            "mode": evaluation_mode(args.limit),
            "limit": args.limit,
            "datasets": args.dataset,
            "rows": variant_rows,
            "average_miou": average_miou,
            "paper_average_miou": paper_average,
            "gap": average_miou - paper_average,
            "relative_percent": average_miou / paper_average * 100.0,
        }
        summary_path = variant_output_dir(args.root, args.output_dir, variant) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        rows.append({"summary": str(summary_path), **summary})
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["group"]), []).append(row)
    payload = {
        "mode": evaluation_mode(args.limit),
        "limit": args.limit,
        "datasets": args.dataset,
        "target_frame": args.target_frame,
        "context_protocol": "common59",
        "alias_policy": args.alias_policy,
        "ignore_zero": args.ignore_zero,
        "vision_layer": args.vision_layer,
        "text_layer": args.text_layer,
        "prompt_templates": TEMPLATES,
        "groups": by_group,
        "rows": rows,
        "protocol": {
            "source_paper_claim": "PAL ablation avg_seg; this file is a corrected-protocol small-sample probe.",
            "known_deviation": known_deviation(args.limit),
            "shared_image_forward": True,
        },
    }
    output_path = args.root / args.output_dir / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {output_path}", flush=True)
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablations/segmentation_probes_fast"))
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS, default=[])
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--full", action="store_true", help="Evaluate the full dataset split instead of the default 64-sample probe.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--vision-model", default="facebook/dinov2-large")
    parser.add_argument("--text-model", default="roberta-large")
    parser.add_argument("--vision-layer", type=int, default=-1)
    parser.add_argument("--text-layer", type=int, default=-2)
    parser.add_argument("--max-text-length", type=int, default=32)
    parser.add_argument("--target-frame", choices=["original", "processor"], default="processor")
    parser.add_argument("--alias-policy", choices=["first", "all", "clean"], default="all")
    parser.add_argument("--ignore-zero", action="store_true", help="Treat label id 0 as void/ignore during mIoU accumulation.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> int:
    args = build_parser().parse_args()
    args.root = args.root.resolve()
    if not args.dataset:
        args.dataset = DEFAULT_DATASETS
    if args.full:
        args.limit = None
    variants = select_variants(args)
    if args.list or not args.run:
        print(json.dumps([
            {
                "name": variant.name,
                "group": variant.group,
                "label": variant.label,
                "checkpoint": str(variant.checkpoint),
                "output_dir": str(variant_output_dir(args.root, args.output_dir, variant)),
            }
            for variant in variants
        ], indent=2, sort_keys=True))
    if args.run:
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        device = resolve_device(args.device)
        print(f"device={device}", flush=True)
        models = {variant.name: load_trained_pal_model(variant.checkpoint, device=device) for variant in variants}
        tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=args.local_files_only)
        text_model = AutoModel.from_pretrained(args.text_model, local_files_only=args.local_files_only).to(device).eval()
        image_processor = AutoImageProcessor.from_pretrained(args.vision_model, local_files_only=args.local_files_only)
        vision_model = AutoModel.from_pretrained(args.vision_model, local_files_only=args.local_files_only).to(device).eval()
        all_results: dict[str, dict[str, dict[str, Any]]] = {}
        for dataset_name in args.dataset:
            all_results[dataset_name] = evaluate_dataset(args, dataset_name, variants, models, tokenizer, text_model, image_processor, vision_model, device)
        write_summaries(args, variants, all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
