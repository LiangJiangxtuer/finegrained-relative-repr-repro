#!/usr/bin/env python3
"""Run a CKA-based encoder layer-pair sweep on Karpathy retrieval samples."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from extract_karpathy_retrieval_tokens import _load_images, build_karpathy_pairs
from pal_repro.cka import rank_layer_pairs

DEFAULT_LAYER_GRID = [-1, -2, -4, -6, -8, -10, -12, -16, -20, -24]
BASELINE_LAYER_PAIR = (-1, -1)


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    weights = mask.to(hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.vision_layer:
        args.vision_layer = list(DEFAULT_LAYER_GRID)
    if not args.text_layer:
        args.text_layer = list(DEFAULT_LAYER_GRID)
    return args


def select_candidate_layer_pairs(
    ranked: list[dict[str, Any]],
    top_k: int,
    baseline: tuple[int, int] = BASELINE_LAYER_PAIR,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = [
        {**row, "selection_reason": "top_cka"} for row in ranked[: max(top_k, 0)]
    ]
    selected_keys = {(row["vision_layer"], row["text_layer"]) for row in selected}
    if baseline not in selected_keys:
        baseline_row = next(
            (
                row for row in ranked
                if (row["vision_layer"], row["text_layer"]) == baseline
            ),
            None,
        )
        if baseline_row is not None:
            selected.append({**baseline_row, "selection_reason": "final_layer_baseline"})
    return selected


@torch.no_grad()
def run_layer_sweep(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs = build_karpathy_pairs(
        karpathy_json=args.karpathy_json,
        dataset=args.dataset,
        split=args.split,
        coco_root=args.coco_root,
        flickr_zip=args.flickr_zip,
        caption_policy=args.caption_policy,
        limit_images=args.limit_images,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=args.local_files_only)
    text_model = AutoModel.from_pretrained(args.text_model, local_files_only=args.local_files_only).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model, local_files_only=args.local_files_only)
    vision_model = AutoModel.from_pretrained(args.vision_model, local_files_only=args.local_files_only).to(device).eval()

    vision_acc: dict[int, list[torch.Tensor]] = defaultdict(list)
    text_acc: dict[int, list[torch.Tensor]] = defaultdict(list)
    zf = zipfile.ZipFile(args.flickr_zip) if args.dataset == "flickr30k" else None
    try:
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            images = _load_images(batch, zf)
            image_inputs = image_processor(images=images, return_tensors="pt").to(device)
            image_outputs = vision_model(**image_inputs, output_hidden_states=True)
            text_inputs = tokenizer(
                [row["caption"] for row in batch],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.max_text_length,
            ).to(device)
            text_outputs = text_model(**text_inputs, output_hidden_states=True)
            for layer in args.vision_layer:
                vision_acc[layer].append(_masked_mean(image_outputs.hidden_states[layer]).detach().cpu())
            for layer in args.text_layer:
                text_acc[layer].append(
                    _masked_mean(text_outputs.hidden_states[layer], text_inputs["attention_mask"]).detach().cpu()
                )
            print(f"processed {min(start + len(batch), len(pairs))}/{len(pairs)}", flush=True)
    finally:
        if zf is not None:
            zf.close()

    vision_layers = {layer: torch.cat(parts, dim=0) for layer, parts in vision_acc.items()}
    text_layers = {layer: torch.cat(parts, dim=0) for layer, parts in text_acc.items()}
    ranked = rank_layer_pairs(vision_layers, text_layers)
    candidate_layer_pairs = select_candidate_layer_pairs(ranked, top_k=args.top_k)
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "caption_policy": args.caption_policy,
        "num_pairs": len(pairs),
        "num_images": len({row["image_id"] for row in pairs}),
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "vision_layers": args.vision_layer,
        "text_layers": args.text_layer,
        "ranking": ranked,
        "top_k": ranked[: args.top_k],
        "candidate_layer_pairs": candidate_layer_pairs,
        "best": ranked[0] if ranked else None,
        "protocol": {
            "source_paper_claim": "Paper states encoder layers are selected using CKA, but exact layer indices are not disclosed.",
            "local_command": " ".join(sys.argv),
            "encoder_layers": {"vision_layers": args.vision_layer, "text_layers": args.text_layer},
            "dataset_split": args.split,
            "prompt_policy": {"caption_policy": args.caption_policy},
            "known_deviation": "This step ranks pooled hidden states only; downstream 10K proxy training and full 80K retraining are separate stages.",
            "verification_status": "implemented",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["coco", "flickr30k"], required=True)
    parser.add_argument("--karpathy-json", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--coco-root", type=Path, default=None)
    parser.add_argument("--flickr-zip", type=Path, default=None)
    parser.add_argument("--caption-policy", choices=["first", "all"], default="first")
    parser.add_argument("--limit-images", type=int, default=1024)
    parser.add_argument("--vision-layer", action="append", type=int, default=[])
    parser.add_argument("--text-layer", action="append", type=int, default=[])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-model", default="facebook/dinov2-large")
    parser.add_argument("--text-model", default="roberta-large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = finalize_args(build_parser().parse_args())
    run_layer_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
