#!/usr/bin/env python3
"""Merge chunked PAL token cache into monolithic .pt files for training.

This is a convenience bridge for `pal_repro.train`, which currently consumes
`image_tokens.pt`, `text_tokens.pt`, and `text_mask.pt`. Run after chunked
extraction if enough CPU RAM is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def merge_kind(root: Path, chunks: list[dict], key: str, output_name: str) -> list[int]:
    parts = [_load(root / chunk[key]) for chunk in chunks]
    merged = torch.cat(parts, dim=0)
    del parts
    torch.save(merged, root / output_name)
    shape = list(merged.shape)
    del merged
    return shape


def merge_chunks(root: Path) -> dict:
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "chunks":
        raise ValueError(f"Expected chunk metadata at {metadata_path}; got format={metadata.get('format')!r}")
    chunks = metadata["chunks"]
    image_shape = merge_kind(root, chunks, "image_tokens", "image_tokens.pt")
    text_shape = merge_kind(root, chunks, "text_tokens", "text_tokens.pt")
    mask_shape = merge_kind(root, chunks, "text_mask", "text_mask.pt")
    merged = dict(metadata)
    merged.update(
        {
            "format": "monolithic_from_chunks",
            "image_tokens": image_shape,
            "text_tokens": text_shape,
            "text_mask": mask_shape,
            "source_chunk_metadata": str(metadata_path),
        }
    )
    (root / "metadata_merged.json").write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge chunked PAL token cache into monolithic tensors.")
    parser.add_argument("token_dir", type=Path)
    args = parser.parse_args()
    manifest = merge_chunks(args.token_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
