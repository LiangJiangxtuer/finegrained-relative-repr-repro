#!/usr/bin/env python3
"""Extract Flickr30k DINOv2/RoBERTa token tensors for PAL retrieval.

The local Flickr30k package is a zip containing `captions.txt`, `Images/*.jpg`,
and duplicated `Images/flickr30k_images/*.jpg`. This script reads directly from
the zip so the 8GB archive does not need to be unpacked.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def build_pairs(
    zip_path: Path,
    captions_member: str,
    image_prefix: str,
    limit: int | None,
    seed: int,
    caption_policy: str = "all",
) -> list[dict[str, Any]]:
    """Build Flickr30k image/caption rows from a zip-local captions CSV."""

    if caption_policy not in {"first", "all"}:
        raise ValueError("caption_policy must be 'first' or 'all'.")
    image_prefix = image_prefix if image_prefix.endswith("/") else image_prefix + "/"
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        with zf.open(captions_member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            captions_by_image: dict[str, list[str]] = {}
            for row in reader:
                file_name = str(row["image"]).strip()
                caption = str(row["caption"]).strip()
                if not file_name or not caption:
                    continue
                member = image_prefix + file_name
                if member not in names:
                    raise FileNotFoundError(f"Missing Flickr30k image member: {member}")
                captions_by_image.setdefault(file_name, []).append(caption)

    file_names = sorted(captions_by_image)
    random.Random(seed).shuffle(file_names)
    if limit is not None:
        file_names = file_names[:limit]

    rows: list[dict[str, Any]] = []
    for file_name in file_names:
        captions = captions_by_image[file_name]
        selected = captions[:1] if caption_policy == "first" else captions
        try:
            image_id = int(Path(file_name).stem)
        except ValueError:
            image_id = abs(hash(file_name))
        for caption_index, caption in enumerate(selected):
            rows.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "zip_member": image_prefix + file_name,
                    "caption": caption,
                    "caption_index": caption_index,
                    "caption_policy": caption_policy,
                    "split": "flickr30k_local",
                }
            )
    return rows


def storage_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported storage dtype: {name}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def flush_chunk(
    output_dir: Path,
    chunk_index: int,
    image_parts: list[torch.Tensor],
    text_parts: list[torch.Tensor],
    mask_parts: list[torch.Tensor],
    start_index: int,
) -> dict[str, Any]:
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    image_tokens = torch.cat(image_parts, dim=0)
    text_tokens = torch.cat(text_parts, dim=0)
    text_mask = torch.cat(mask_parts, dim=0)
    prefix = f"chunk_{chunk_index:05d}"
    image_path = chunk_dir / f"{prefix}_image_tokens.pt"
    text_path = chunk_dir / f"{prefix}_text_tokens.pt"
    mask_path = chunk_dir / f"{prefix}_text_mask.pt"
    torch.save(image_tokens, image_path)
    torch.save(text_tokens, text_path)
    torch.save(text_mask, mask_path)
    return {
        "chunk_index": chunk_index,
        "start": start_index,
        "end": start_index + int(image_tokens.shape[0]),
        "num_samples": int(image_tokens.shape[0]),
        "image_tokens": str(image_path.relative_to(output_dir)),
        "text_tokens": str(text_path.relative_to(output_dir)),
        "text_mask": str(mask_path.relative_to(output_dir)),
        "image_shape": list(image_tokens.shape),
        "text_shape": list(text_tokens.shape),
        "mask_shape": list(text_mask.shape),
    }


def select_hidden_state(outputs: Any, layer: int | None) -> torch.Tensor:
    """Return last_hidden_state or a requested hidden-state layer."""

    if layer is None:
        return outputs.last_hidden_state
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model output did not include hidden_states; pass output_hidden_states=True.")
    return hidden_states[layer]


def _load_images_from_zip(zf: zipfile.ZipFile, batch: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for row in batch:
        with zf.open(row["zip_member"]) as handle:
            payload = handle.read()
        images.append(Image.open(io.BytesIO(payload)).convert("RGB"))
    return images


@torch.no_grad()
def extract_tokens(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs = build_pairs(
        zip_path=args.zip_path,
        captions_member=args.captions_member,
        image_prefix=args.image_prefix,
        limit=args.limit,
        seed=args.seed,
        caption_policy=args.caption_policy,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = storage_dtype(args.storage_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model, local_files_only=args.local_files_only)
    text_model = AutoModel.from_pretrained(args.text_model, local_files_only=args.local_files_only).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model, local_files_only=args.local_files_only)
    vision_model = AutoModel.from_pretrained(args.vision_model, local_files_only=args.local_files_only).to(device).eval()

    image_parts: list[torch.Tensor] = []
    text_parts: list[torch.Tensor] = []
    mask_parts: list[torch.Tensor] = []
    chunks: list[dict[str, Any]] = []
    chunk_start = 0
    chunk_index = 0
    samples_in_buffer = 0

    with zipfile.ZipFile(args.zip_path) as zf:
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            captions = [row["caption"] for row in batch]
            text_inputs = tokenizer(
                captions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.max_text_length,
            ).to(device)
            text_outputs = text_model(
                **text_inputs,
                output_hidden_states=args.text_layer is not None,
            )
            text_parts.append(select_hidden_state(text_outputs, args.text_layer).detach().cpu().to(dtype=dtype))
            mask_parts.append(text_inputs["attention_mask"].detach().cpu().bool())

            images = _load_images_from_zip(zf, batch)
            image_inputs = image_processor(images=images, return_tensors="pt").to(device)
            image_outputs = vision_model(
                **image_inputs,
                output_hidden_states=args.vision_layer is not None,
            )
            image_parts.append(select_hidden_state(image_outputs, args.vision_layer).detach().cpu().to(dtype=dtype))

            samples_in_buffer += len(batch)
            processed = min(start + len(batch), len(pairs))
            print(f"processed {processed}/{len(pairs)}", flush=True)
            if samples_in_buffer >= args.chunk_size:
                chunks.append(flush_chunk(output_dir, chunk_index, image_parts, text_parts, mask_parts, chunk_start))
                chunk_start += samples_in_buffer
                chunk_index += 1
                samples_in_buffer = 0
                image_parts.clear()
                text_parts.clear()
                mask_parts.clear()

    if samples_in_buffer:
        chunks.append(flush_chunk(output_dir, chunk_index, image_parts, text_parts, mask_parts, chunk_start))
    pairs_path = output_dir / "pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    manifest = {
        "format": "chunks",
        "dataset": "Flickr30k",
        "zip_path": str(args.zip_path),
        "captions_member": args.captions_member,
        "image_prefix": args.image_prefix,
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "vision_layer": args.vision_layer,
        "text_layer": args.text_layer,
        "caption_policy": args.caption_policy,
        "num_pairs": len(pairs),
        "num_images": len({row["image_id"] for row in pairs}),
        "storage_dtype": args.storage_dtype,
        "max_text_length": args.max_text_length,
        "pairs_jsonl": str(pairs_path),
        "chunks": chunks,
    }
    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--captions-member", default="captions.txt")
    parser.add_argument("--image-prefix", default="Images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caption-policy", choices=["first", "all"], default="all")
    parser.add_argument("--limit", type=int, default=1000, help="Limit selected images before applying caption policy.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vision-model", default="facebook/dinov2-large")
    parser.add_argument("--text-model", default="roberta-large")
    parser.add_argument("--vision-layer", type=int, default=None)
    parser.add_argument("--text-layer", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = extract_tokens(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
