#!/usr/bin/env python3
"""Evaluate PAL zero-shot classification on paper datasets.

This runner keeps the frozen DINOv2/RoBERTa encoders fixed, restores the trained
PAL anchor checkpoint, encodes class prompts with RoBERTa, encodes images with
DINOv2, and reports top-1/top-5 accuracy from PAL-relative profiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from pal_repro.classification import build_class_prompts, classification_topk
from pal_repro.evaluate import load_trained_pal_model

DEFAULT_ROOTS = {
    "CIFAR100": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/cifar100"),
    "STL10": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/stl10"),
    "Caltech101": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/caltech101"),
    "DTD": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/dtd"),
    "EuroSAT": Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/pal_public/classification/eurosat"),
}


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


@torch.no_grad()
def encode_class_texts(
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
    dataset, class_names, split = load_dataset(args.dataset, root)
    if args.limit is not None:
        indices = list(range(min(args.limit, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pil,
    )

    pal_model = load_trained_pal_model(args.checkpoint, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=args.local_files_only)
    text_model = AutoModel.from_pretrained(args.text_model, local_files_only=args.local_files_only).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model, local_files_only=args.local_files_only)
    vision_model = AutoModel.from_pretrained(args.vision_model, local_files_only=args.local_files_only).to(device).eval()

    prompts = build_class_prompts(class_names, template=args.prompt_template)
    class_features = encode_class_texts(
        pal_model,
        tokenizer,
        text_model,
        prompts,
        device=device,
        max_length=args.max_text_length,
    )
    class_features = F.normalize(class_features, dim=-1)

    pred_scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    processed = 0
    for images, batch_labels in loader:
        image_inputs = image_processor(images=images, return_tensors="pt").to(device)
        image_outputs = vision_model(**image_inputs)
        image_features = pal_model.encode_image(image_outputs.last_hidden_state.float()).detach().cpu()
        similarity = F.normalize(image_features, dim=-1) @ class_features.T
        pred_scores.append(similarity)
        labels.append(batch_labels.cpu())
        processed += len(images)
        print(f"processed {processed}/{len(dataset)}", flush=True)

    similarity_all = torch.cat(pred_scores, dim=0)
    labels_all = torch.cat(labels, dim=0)
    metrics = classification_topk(similarity_all, labels_all, ks=(1, 5))
    result: dict[str, Any] = {
        "dataset": args.dataset,
        "split": split,
        "root": str(root),
        "checkpoint": str(args.checkpoint),
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "num_samples": int(labels_all.shape[0]),
        "num_classes": len(class_names),
        "prompt_template": args.prompt_template,
        "prompts": prompts,
        "batch_size": args.batch_size,
        "device": str(device),
        "metrics": metrics,
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
    parser.add_argument("--batch-size", type=int, default=64)
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
