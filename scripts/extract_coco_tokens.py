#!/usr/bin/env python3
"""Extract COCO2014 DINOv2/RoBERTa token tensors for strict PAL.

For paper-scale extraction, prefer chunked fp16 output:

```bash
python scripts/extract_coco_tokens.py \
  --captions-json .../captions_train2014.json \
  --image-dir .../train2014 \
  --output-dir data/tokens/coco2014_full \
  --output-format chunks --chunk-size 2048 --storage-dtype float16
```

The monolithic format writes `image_tokens.pt`, `text_tokens.pt`, and
`text_mask.pt`; chunked format writes `chunks/chunk_XXXXX_*.pt` plus
`metadata.json` and is safer for the full 80K COCO split.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def build_pairs(
    captions_json: Path,
    image_dir: Path,
    limit: int | None,
    seed: int,
    caption_policy: str = "first",
) -> list[dict[str, Any]]:
    data = json.loads(captions_json.read_text(encoding="utf-8"))
    if caption_policy not in {"first", "all"}:
        raise ValueError("caption_policy must be 'first' or 'all'.")
    image_by_id = {int(item["id"]): item["file_name"] for item in data["images"]}
    captions_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in image_by_id:
            continue
        captions_by_image.setdefault(image_id, []).append(
            {
                "caption": str(ann["caption"]),
                "annotation_id": int(ann["id"]),
                "caption_index": len(captions_by_image.get(image_id, [])),
            }
        )
    image_ids = sorted(captions_by_image)
    random.Random(seed).shuffle(image_ids)
    if limit is not None:
        image_ids = image_ids[:limit]
    pairs: list[dict[str, Any]] = []
    for image_id in image_ids:
        file_name = image_by_id[image_id]
        image_path = image_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing COCO image: {image_path}")
        captions = captions_by_image[image_id]
        selected_captions = captions[:1] if caption_policy == "first" else captions
        for caption in selected_captions:
            pairs.append(
                {
                    "image_id": image_id,
                    "annotation_id": caption["annotation_id"],
                    "caption_index": caption["caption_index"],
                    "caption_policy": caption_policy,
                    "file_name": file_name,
                    "image_path": str(image_path),
                    "caption": caption["caption"],
                }
            )
    return pairs


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


@torch.no_grad()
def extract_tokens(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    if args.output_format == "chunks" and args.dynamic_padding:
        raise ValueError("Chunked output requires fixed padding; omit --dynamic-padding.")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs = build_pairs(
        args.captions_json,
        args.image_dir,
        args.limit,
        args.seed,
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

    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        captions = [row["caption"] for row in batch]
        text_inputs = tokenizer(
            captions,
            return_tensors="pt",
            padding=True if args.dynamic_padding else "max_length",
            truncation=True,
            max_length=args.max_text_length,
        ).to(device)
        text_outputs = text_model(**text_inputs)
        text_parts.append(text_outputs.last_hidden_state.detach().cpu().to(dtype=dtype))
        mask_parts.append(text_inputs["attention_mask"].detach().cpu().bool())

        images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
        image_inputs = image_processor(images=images, return_tensors="pt").to(device)
        image_outputs = vision_model(**image_inputs)
        image_parts.append(image_outputs.last_hidden_state.detach().cpu().to(dtype=dtype))

        samples_in_buffer += len(batch)
        processed = min(start + len(batch), len(pairs))
        print(f"processed {processed}/{len(pairs)}", flush=True)

        if args.output_format == "chunks" and samples_in_buffer >= args.chunk_size:
            chunks.append(flush_chunk(output_dir, chunk_index, image_parts, text_parts, mask_parts, chunk_start))
            chunk_start += samples_in_buffer
            chunk_index += 1
            samples_in_buffer = 0
            image_parts.clear()
            text_parts.clear()
            mask_parts.clear()

    if args.output_format == "chunks":
        if samples_in_buffer:
            chunks.append(flush_chunk(output_dir, chunk_index, image_parts, text_parts, mask_parts, chunk_start))
        manifest = {
            "format": "chunks",
            "vision_model": args.vision_model,
            "text_model": args.text_model,
            "num_pairs": len(pairs),
            "caption_policy": args.caption_policy,
            "storage_dtype": args.storage_dtype,
            "max_text_length": args.max_text_length,
            "captions_json": str(args.captions_json),
            "image_dir": str(args.image_dir),
            "chunks": chunks,
        }
    else:
        image_tokens = torch.cat(image_parts, dim=0)
        text_tokens = torch.cat(text_parts, dim=0)
        text_mask = torch.cat(mask_parts, dim=0)
        torch.save(image_tokens, output_dir / "image_tokens.pt")
        torch.save(text_tokens, output_dir / "text_tokens.pt")
        torch.save(text_mask, output_dir / "text_mask.pt")
        manifest = {
            "format": "monolithic",
            "vision_model": args.vision_model,
            "text_model": args.text_model,
            "num_pairs": len(pairs),
            "caption_policy": args.caption_policy,
            "storage_dtype": args.storage_dtype,
            "image_tokens": list(image_tokens.shape),
            "text_tokens": list(text_tokens.shape),
            "text_mask": list(text_mask.shape),
            "captions_json": str(args.captions_json),
            "image_dir": str(args.image_dir),
        }

    pairs_path = output_dir / "pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    manifest["pairs_jsonl"] = str(pairs_path)
    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract COCO2014 token tensors for PAL.")
    parser.add_argument("--captions-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vision-model", default="facebook/dinov2-large")
    parser.add_argument("--text-model", default="roberta-large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Limit selected images before applying caption policy.")
    parser.add_argument("--caption-policy", choices=["first", "all"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dynamic-padding", action="store_true", help="Use per-batch dynamic text padding; not valid with chunks.")
    parser.add_argument("--pad-to-max-length", action="store_true", help="Deprecated; fixed max-length padding is now the default.")
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--output-format", choices=["monolithic", "chunks"], default="monolithic")
    parser.add_argument("--chunk-size", type=int, default=2048)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = extract_tokens(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
