#!/usr/bin/env python3
"""Extract Karpathy-split COCO/Flickr30k retrieval tokens for PAL.

The paper's retrieval numbers are normally reported on the standard Karpathy
1K/5K evaluation splits. This script consumes `dataset_coco.json` or
`dataset_flickr30k.json` from Karpathy's caption dataset metadata and writes the
same chunked fp16 token-cache format used by the rest of the reproduction.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sentence_text(sentence: dict[str, Any]) -> str:
    text = sentence.get("raw") or sentence.get("sent") or sentence.get("tokens")
    if isinstance(text, list):
        return " ".join(str(item) for item in text)
    if text is None:
        raise ValueError(f"Karpathy sentence is missing raw text: {sentence}")
    return str(text)


def _image_id(image: dict[str, Any], fallback_index: int) -> int:
    if "imgid" in image:
        return int(image["imgid"])
    if "cocoid" in image:
        return int(image["cocoid"])
    stem = Path(str(image.get("filename", fallback_index))).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else int(fallback_index)


def _resolve_coco_image_path(coco_root: Path, image: dict[str, Any]) -> Path:
    filename = str(image["filename"])
    filepath = image.get("filepath")
    candidates: list[Path] = []
    if filepath:
        candidates.append(coco_root / str(filepath) / filename)
    candidates.extend(
        [
            coco_root / filename,
            coco_root / "val2014" / filename,
            coco_root / "train2014" / filename,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve COCO image {filename!r} under {coco_root}; tried "
        + ", ".join(str(item) for item in candidates)
    )


def _resolve_flickr_zip_member(zf: zipfile.ZipFile, filename: str) -> str:
    candidates = [
        f"Images/{filename}",
        f"Images/flickr30k_images/{filename}",
        filename,
    ]
    names = set(zf.namelist())
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise FileNotFoundError(f"Could not resolve Flickr30k image {filename!r} in {zf.filename}")


def build_karpathy_pairs(
    karpathy_json: Path,
    dataset: str,
    split: str,
    coco_root: Path | None,
    flickr_zip: Path | None,
    caption_policy: str = "all",
    limit_images: int | None = None,
) -> list[dict[str, Any]]:
    """Build retrieval rows from a Karpathy split JSON file.

    Each output row represents one image-caption pair. Multi-caption retrieval
    duplicates the image path/member across captions, matching the current PAL
    token-cache/evaluation contract.
    """

    if caption_policy not in {"first", "all"}:
        raise ValueError("caption_policy must be 'first' or 'all'.")
    if dataset not in {"coco", "flickr30k"}:
        raise ValueError("dataset must be 'coco' or 'flickr30k'.")
    if dataset == "coco" and coco_root is None:
        raise ValueError("coco_root is required for COCO Karpathy extraction.")
    if dataset == "flickr30k" and flickr_zip is None:
        raise ValueError("flickr_zip is required for Flickr30k Karpathy extraction.")

    payload = _read_json(karpathy_json)
    selected = [image for image in payload["images"] if str(image.get("split")) == split]
    if limit_images is not None:
        selected = selected[:limit_images]

    rows: list[dict[str, Any]] = []
    zip_members: dict[str, str] = {}
    if dataset == "flickr30k":
        assert flickr_zip is not None
        with zipfile.ZipFile(flickr_zip) as zf:
            for image in selected:
                filename = str(image["filename"])
                zip_members[filename] = _resolve_flickr_zip_member(zf, filename)

    for image_index, image in enumerate(selected):
        image_id = _image_id(image, fallback_index=image_index)
        filename = str(image["filename"])
        sentences = list(image.get("sentences", []))
        chosen = sentences[:1] if caption_policy == "first" else sentences
        if dataset == "coco":
            assert coco_root is not None
            image_path = str(_resolve_coco_image_path(coco_root, image))
            zip_member = None
        else:
            image_path = None
            zip_member = zip_members[filename]
        for caption_index, sentence in enumerate(chosen):
            rows.append(
                {
                    "dataset": dataset,
                    "karpathy_split": split,
                    "caption_policy": caption_policy,
                    "image_id": image_id,
                    "file_name": filename,
                    "image_path": image_path,
                    "zip_member": zip_member,
                    "caption": _sentence_text(sentence),
                    "caption_index": caption_index,
                    "sentence_id": sentence.get("sentid"),
                    "source_filepath": image.get("filepath"),
                }
            )
    if not rows:
        raise ValueError(f"No Karpathy rows selected for dataset={dataset}, split={split}.")
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


def _select_hidden_state(outputs: Any, layer: int | None) -> torch.Tensor:
    if layer is None:
        return outputs.last_hidden_state
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model output did not include hidden_states; pass output_hidden_states=True.")
    return hidden_states[layer]


def _load_images(rows: list[dict[str, Any]], zf: zipfile.ZipFile | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    for row in rows:
        if zf is not None:
            with zf.open(str(row["zip_member"])) as handle:
                payload = handle.read()
            images.append(Image.open(io.BytesIO(payload)).convert("RGB"))
        else:
            images.append(Image.open(str(row["image_path"])).convert("RGB"))
    return images


@torch.no_grad()
def extract_tokens(args: argparse.Namespace) -> dict[str, Any]:
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

    zf_context = zipfile.ZipFile(args.flickr_zip) if args.dataset == "flickr30k" else None
    try:
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
            text_parts.append(_select_hidden_state(text_outputs, args.text_layer).detach().cpu().to(dtype=dtype))
            mask_parts.append(text_inputs["attention_mask"].detach().cpu().bool())

            images = _load_images(batch, zf_context)
            image_inputs = image_processor(images=images, return_tensors="pt").to(device)
            image_outputs = vision_model(
                **image_inputs,
                output_hidden_states=args.vision_layer is not None,
            )
            image_parts.append(_select_hidden_state(image_outputs, args.vision_layer).detach().cpu().to(dtype=dtype))

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
    finally:
        if zf_context is not None:
            zf_context.close()

    if samples_in_buffer:
        chunks.append(flush_chunk(output_dir, chunk_index, image_parts, text_parts, mask_parts, chunk_start))

    pairs_path = output_dir / "pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    manifest = {
        "format": "chunks",
        "dataset": args.dataset,
        "karpathy_json": str(args.karpathy_json),
        "split": args.split,
        "caption_policy": args.caption_policy,
        "num_pairs": len(pairs),
        "num_images": len({row["image_id"] for row in pairs}),
        "coco_root": None if args.coco_root is None else str(args.coco_root),
        "flickr_zip": None if args.flickr_zip is None else str(args.flickr_zip),
        "vision_model": args.vision_model,
        "text_model": args.text_model,
        "vision_layer": args.vision_layer,
        "text_layer": args.text_layer,
        "storage_dtype": args.storage_dtype,
        "max_text_length": args.max_text_length,
        "pairs_jsonl": str(pairs_path),
        "chunks": chunks,
    }
    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["coco", "flickr30k"], required=True)
    parser.add_argument("--karpathy-json", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--coco-root", type=Path, default=None)
    parser.add_argument("--flickr-zip", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caption-policy", choices=["first", "all"], default="all")
    parser.add_argument("--limit-images", type=int, default=None)
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
