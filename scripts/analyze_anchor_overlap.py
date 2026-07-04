#!/usr/bin/env python3
"""Compute PAL top-k anchor overlap/Dice metrics on cached retrieval tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pal_repro.analysis import anchor_overlap_report
from pal_repro.evaluate import _encode_token_rows, load_trained_pal_model


def analyze_anchor_overlap(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_trained_pal_model(args.checkpoint, device=device)
    image_features, text_features, dim_img, dim_txt = _encode_token_rows(
        model=model,
        token_dir=args.token_dir,
        batch_size=args.batch_size,
        device=device,
    )
    if args.limit is not None:
        image_features = image_features[: args.limit]
        text_features = text_features[: args.limit]
    report = anchor_overlap_report(image_features, text_features, k=args.top_k)
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "token_dir": str(args.token_dir),
        "num_pairs": int(image_features.shape[0]),
        "dim_img": dim_img,
        "dim_txt": dim_txt,
        "top_k": args.top_k,
        "metrics": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    analyze_anchor_overlap(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
