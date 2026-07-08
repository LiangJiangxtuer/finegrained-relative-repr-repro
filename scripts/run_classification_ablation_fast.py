#!/usr/bin/env python3
"""Fast multi-checkpoint zero-shot classification ablation for PAL.

This evaluates all trained PAL ablation checkpoints while sharing frozen DINOv2
image forwards per dataset/batch. It is much faster than invoking the single
checkpoint evaluator once per checkpoint because image tokens are independent of
PAL anchors.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from pal_repro.classification import build_class_prompt_groups
from pal_repro.evaluate import load_trained_pal_model

DEFAULT_ROOTS = {
    "CIFAR100": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/cifar100"),
    "STL10": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/stl10"),
    "Caltech101": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/caltech101"),
    "DTD": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/dtd"),
    "EuroSAT": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/eurosat"),
}
DEFAULT_DATASETS = ["STL10", "CIFAR100", "Caltech101", "DTD", "EuroSAT"]
TEMPLATES = [
    "a photo of {class_name}",
    "a cropped photo of {class_name}",
    "a close-up photo of {class_name}",
    "a clean photo of {class_name}",
]
PAPER_AVG_TOP1 = 51.46


@dataclass(frozen=True)
class Variant:
    group: str
    label: str
    checkpoint: Path

    @property
    def name(self) -> str:
        return f"{self.group}_{self.label}"


@dataclass
class RunningMetrics:
    correct1: int
    correct5: int
    total: int
    per_class_correct: list[int]
    per_class_total: list[int]

    @classmethod
    def create(cls, num_classes: int) -> "RunningMetrics":
        return cls(0, 0, 0, [0] * num_classes, [0] * num_classes)

    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        max_k = min(5, scores.shape[1])
        top = scores.topk(max_k, dim=1).indices
        top1 = top[:, 0]
        self.correct1 += int((top1 == labels).sum().item())
        self.correct5 += int((top == labels[:, None]).any(dim=1).sum().item())
        self.total += int(labels.numel())
        for class_idx in labels.unique().tolist():
            mask = labels == int(class_idx)
            self.per_class_total[int(class_idx)] += int(mask.sum().item())
            self.per_class_correct[int(class_idx)] += int((top1[mask] == labels[mask]).sum().item())

    def payload(self, class_names: list[str]) -> tuple[dict[str, float], dict[str, float | None]]:
        top1 = self.correct1 / max(self.total, 1) * 100.0
        top5 = self.correct5 / max(self.total, 1) * 100.0
        per_class: dict[str, float | None] = {}
        for idx, class_name in enumerate(class_names):
            if self.per_class_total[idx] == 0:
                per_class[class_name] = None
            else:
                per_class[class_name] = self.per_class_correct[idx] / self.per_class_total[idx] * 100.0
        return {"top1": top1, "top5": top5}, per_class


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


def load_dataset(name: str, root: Path):
    if name == "CIFAR100":
        ds = datasets.CIFAR100(root=str(root), train=False, download=False)
        return ds, list(ds.classes), "test"
    if name == "STL10":
        ds = datasets.STL10(root=str(root), split="test", download=False)
        return ds, list(ds.classes), "test"
    if name == "Caltech101":
        ds = datasets.Caltech101(root=str(root), download=False)
        return ds, list(ds.categories), "all"
    if name == "DTD":
        ds = datasets.DTD(root=str(root), split="test", partition=1, download=False)
        return ds, list(ds.classes), "test1"
    if name == "EuroSAT":
        ds = datasets.EuroSAT(root=str(root), download=False)
        return ds, list(ds.classes), "all"
    raise ValueError(f"Unsupported dataset: {name}")


def collate_pil(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def variant_output_dir(root: Path, output_dir: Path, variant: Variant) -> Path:
    return root / output_dir / variant.group / variant.label


def dataset_output_path(root: Path, output_dir: Path, variant: Variant, dataset: str) -> Path:
    return variant_output_dir(root, output_dir, variant) / "ensemble" / f"{dataset.lower()}_classification.json"


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
    class_names: list[str],
    device: torch.device,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], list[list[str]], list[str]]:
    prompt_groups = build_class_prompt_groups(class_names, templates=TEMPLATES)
    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    inputs = tokenizer(
        flat_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = text_model(**inputs)
    text_tokens = outputs.last_hidden_state.float()
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
    return features, prompt_groups, flat_prompts


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
    root = args.dataset_root.get(dataset_name) if hasattr(args, "dataset_root") else None
    dataset, class_names, split = load_dataset(dataset_name, Path(root) if root else DEFAULT_ROOTS[dataset_name])
    if args.limit is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_pil)
    class_features, prompt_groups, prompts = encode_class_features(
        variants,
        models,
        tokenizer,
        text_model,
        class_names,
        device=device,
        max_length=args.max_text_length,
    )
    metrics = {variant.name: RunningMetrics.create(len(class_names)) for variant in variants}
    processed = 0
    for images, labels_cpu in loader:
        image_inputs = image_processor(images=images, return_tensors="pt").to(device)
        image_tokens = vision_model(**image_inputs).last_hidden_state.float()
        labels = labels_cpu.to(device)
        for variant in variants:
            model = models[variant.name]
            image_features = F.normalize(model.encode_image(image_tokens), dim=-1)
            scores = image_features @ class_features[variant.name].T
            metrics[variant.name].update(scores, labels)
        processed += len(images)
        print(f"{dataset_name}: processed {processed}/{len(dataset)}", flush=True)
    results: dict[str, dict[str, Any]] = {}
    for variant in variants:
        metric_payload, per_class = metrics[variant.name].payload(class_names)
        output = dataset_output_path(args.root, args.output_dir, variant, dataset_name)
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
            "num_samples": metrics[variant.name].total,
            "num_classes": len(class_names),
            "prompt_templates": TEMPLATES,
            "prompt_groups": prompt_groups,
            "prompts": prompts,
            "batch_size": args.batch_size,
            "device": str(device),
            "metrics": metric_payload,
            "per_class_accuracy": per_class,
            "protocol": {
                "source_paper_claim": "PAL ablation tables report zero-shot classification top-1.",
                "dataset_split": split,
                "prompt_policy": {"mode": "fixed_ensemble", "templates": TEMPLATES},
                "known_deviation": "Uses repository fixed fair prompt ensemble; exact paper prompt set is not fully specified.",
                "shared_image_forward": True,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        results[variant.name] = result
    return results


def write_variant_summaries(args: argparse.Namespace, variants: list[Variant], all_results: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = []
        for dataset_name in args.dataset:
            result = all_results[dataset_name][variant.name]
            variant_rows.append({
                "dataset": dataset_name,
                "num_samples": result["num_samples"],
                "output": str(dataset_output_path(args.root, args.output_dir, variant, dataset_name)),
                "templates": TEMPLATES,
                "top1": result["metrics"]["top1"],
                "top5": result["metrics"].get("top5"),
            })
        average_top1 = sum(float(row["top1"]) for row in variant_rows) / max(len(variant_rows), 1)
        summary = {
            "checkpoint": str(variant.checkpoint),
            "variant": variant.name,
            "group": variant.group,
            "label": variant.label,
            "mode": "shared_image_forward_fixed_prompt_ensemble",
            "datasets": args.dataset,
            "templates": TEMPLATES,
            "rows": variant_rows,
            "average_top1": average_top1,
            "paper_average_top1": PAPER_AVG_TOP1,
            "gap": average_top1 - PAPER_AVG_TOP1,
            "relative_percent": average_top1 / PAPER_AVG_TOP1 * 100.0,
        }
        summary_path = variant_output_dir(args.root, args.output_dir, variant) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        rows.append({"summary": str(summary_path), **summary})
    return rows


def write_combined_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["group"]), []).append(row)
    payload = {
        "mode": "shared_image_forward_fixed_prompt_ensemble_classification_ablation",
        "prompt_templates": TEMPLATES,
        "paper_average_top1": PAPER_AVG_TOP1,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "datasets": args.dataset,
        "groups": by_group,
        "rows": rows,
        "protocol": {
            "source_paper_claim": "PAL ablation tables report average zero-shot classification top-1.",
            "dataset_split": "torchvision/default test split per dataset loader",
            "prompt_policy": {"mode": "fixed_ensemble", "templates": TEMPLATES},
            "known_deviation": "Uses repository fixed fair prompt ensemble; exact paper prompt set is not fully specified.",
            "shared_image_forward": True,
        },
    }
    output_path = args.root / args.output_dir / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {output_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablations/classification_fast"))
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS, default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", action="append", default=[], help="Variant name such as k_32, tau_0_02, or token_usage_cap.")
    parser.add_argument("--vision-model", default="facebook/dinov2-large")
    parser.add_argument("--text-model", default="roberta-large")
    parser.add_argument("--max-text-length", type=int, default=32)
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
            all_results[dataset_name] = evaluate_dataset(
                args,
                dataset_name,
                variants,
                models,
                tokenizer,
                text_model,
                image_processor,
                vision_model,
                device,
            )
        rows = write_variant_summaries(args, variants, all_results)
        write_combined_summary(args, rows)
        print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
