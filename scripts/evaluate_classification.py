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

from pal_repro.classification import (
    build_class_prompt_groups,
    build_class_prompts,
    classification_topk,
    per_class_accuracy,
)
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


def select_hidden_state(outputs: Any, layer: int | None) -> torch.Tensor:
    """Return last_hidden_state or a requested hidden-state layer."""

    if layer is None:
        return outputs.last_hidden_state
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model output did not include hidden_states; pass output_hidden_states=True.")
    return hidden_states[layer]


@torch.no_grad()
def encode_class_prompt_groups(
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
    flat_features = F.normalize(
        pal_model.encode_text(text_tokens, inputs["attention_mask"].bool()).detach().cpu(),
        dim=-1,
    )
    grouped: list[torch.Tensor] = []
    offset = 0
    for group in prompt_groups:
        width = len(group)
        grouped.append(F.normalize(flat_features[offset:offset + width].mean(dim=0), dim=0))
        offset += width
    return torch.stack(grouped, dim=0)


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

    templates = args.prompt_template or ["a photo of {class_name}"]
    prompt_groups = build_class_prompt_groups(class_names, templates=templates)
    prompts = [prompt for group in prompt_groups for prompt in group]
    class_features = encode_class_prompt_groups(
        pal_model,
        tokenizer,
        text_model,
        prompt_groups,
        device=device,
        max_length=args.max_text_length,
        text_layer=args.text_layer,
    )
    class_features = F.normalize(class_features, dim=-1)

    pred_scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    processed = 0
    for images, batch_labels in loader:
        image_inputs = image_processor(images=images, return_tensors="pt").to(device)
        image_outputs = vision_model(**image_inputs, output_hidden_states=args.vision_layer is not None)
        image_tokens = select_hidden_state(image_outputs, args.vision_layer).float()
        image_features = pal_model.encode_image(image_tokens).detach().cpu()
        similarity = F.normalize(image_features, dim=-1) @ class_features.T
        pred_scores.append(similarity)
        labels.append(batch_labels.cpu())
        processed += len(images)
        print(f"processed {processed}/{len(dataset)}", flush=True)

    similarity_all = torch.cat(pred_scores, dim=0)
    labels_all = torch.cat(labels, dim=0)
    metrics = classification_topk(similarity_all, labels_all, ks=(1, 5))
    class_accuracy = per_class_accuracy(similarity_all, labels_all, class_names)
    result: dict[str, Any] = {
        "dataset": args.dataset,
        "split": split,
        "root": str(root),
        "checkpoint": str(args.checkpoint),
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "vision_layer": args.vision_layer,
        "text_layer": args.text_layer,
        "num_samples": int(labels_all.shape[0]),
        "num_classes": len(class_names),
        "prompt_template": templates[0] if len(templates) == 1 else None,
        "prompt_templates": templates,
        "prompt_groups": prompt_groups,
        "prompts": prompts,
        "batch_size": args.batch_size,
        "device": str(device),
        "metrics": metrics,
        "per_class_accuracy": class_accuracy,
        "protocol": {
            "source_paper_claim": "Table 2 zero-shot classification top-1",
            "encoder_layers": {"vision_layer": args.vision_layer, "text_layer": args.text_layer},
            "dataset_split": split,
            "prompt_policy": {"templates": templates, "aggregation": "normalized_mean_per_class"},
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-text-length", type=int, default=32)
    parser.add_argument("--prompt-template", action="append", default=None)
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
