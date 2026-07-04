"""Command-line evaluation helpers for trained PAL checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pal_repro.data import load_token_tensors
from pal_repro.eval import (
    encode_images_batched,
    encode_texts_batched,
    evaluate_retrieval_model,
    multicaption_retrieval_metrics,
    retrieval_metrics,
)
from pal_repro.models.pal import ProjectionFreeAnchorLearning, pal_trainable_parameter_names


def load_trained_pal_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> ProjectionFreeAnchorLearning:
    """Restore a strict PAL model from a reproduction checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    anchors_img = state["anchors_img"]
    anchors_txt = state["anchors_txt"]
    config = checkpoint.get("config", {})
    num_anchors = int(config.get("num_anchors", anchors_img.shape[0]))
    pool_temperature = float(config.get("pool_temperature", 0.03))
    pooling_mode = str(config.get("pooling_mode", "cap"))
    model = ProjectionFreeAnchorLearning(
        dim_img=int(checkpoint.get("dim_img", anchors_img.shape[1])),
        dim_txt=int(checkpoint.get("dim_txt", anchors_txt.shape[1])),
        num_anchors=num_anchors,
        pool_temperature=pool_temperature,
        pooling_mode=pooling_mode,
    )
    model.load_state_dict(state)
    model.to(resolved_device)
    model.eval()
    return model


def evaluate_retrieval_token_dir(
    checkpoint_path: str | Path,
    token_dir: str | Path,
    output_path: str | Path,
    batch_size: int = 256,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Evaluate one-caption-per-image retrieval on a token tensor directory."""

    token_dir = Path(token_dir)
    output_path = Path(output_path)
    model = load_trained_pal_model(checkpoint_path, device=device)
    token_format = _token_dir_format(token_dir)
    if token_format == "chunks":
        payload = _evaluate_retrieval_chunked(
            model=model,
            token_dir=token_dir,
            batch_size=batch_size,
            device=_resolve_device(device),
        )
    else:
        tensors = load_token_tensors(token_dir, map_location="cpu")
        metrics = evaluate_retrieval_model(
            model,
            tensors.image_tokens,
            tensors.text_tokens,
            tensors.text_mask,
            batch_size=batch_size,
            device=_resolve_device(device),
        )
        payload = {
            "token_format": token_format,
            "num_samples": tensors.num_samples,
            "dim_img": tensors.dim_img,
            "dim_txt": tensors.dim_txt,
            "metrics": metrics,
        }
    result: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint_path)),
        "token_dir": str(token_dir),
        "batch_size": batch_size,
        "device": str(_resolve_device(device)),
        "parameter_names": pal_trainable_parameter_names(model),
        **payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def evaluate_multicaption_retrieval_token_dir(
    checkpoint_path: str | Path,
    token_dir: str | Path,
    output_path: str | Path,
    batch_size: int = 256,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Evaluate COCO/Flickr-style retrieval with multiple captions per image."""

    token_dir = Path(token_dir)
    output_path = Path(output_path)
    resolved_device = _resolve_device(device)
    model = load_trained_pal_model(checkpoint_path, device=resolved_device)
    image_features, text_features, dim_img, dim_txt = _encode_token_rows(
        model=model,
        token_dir=token_dir,
        batch_size=batch_size,
        device=resolved_device,
    )
    rows = _read_pairs_jsonl(_pairs_jsonl_path(token_dir))
    if len(rows) != text_features.shape[0]:
        raise ValueError(
            f"pairs_jsonl rows ({len(rows)}) must match text rows ({text_features.shape[0]})."
        )
    row_image_ids = [int(row["image_id"]) for row in rows]
    first_index_by_image: dict[int, int] = {}
    for idx, image_id in enumerate(row_image_ids):
        first_index_by_image.setdefault(image_id, idx)
    unique_image_ids = list(first_index_by_image)
    unique_indices = torch.tensor(list(first_index_by_image.values()), dtype=torch.long)
    unique_image_features = image_features.index_select(0, unique_indices)
    metrics = multicaption_retrieval_metrics(
        unique_image_features,
        text_features,
        image_ids=unique_image_ids,
        text_image_ids=row_image_ids,
        percent=True,
    )
    result: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint_path)),
        "token_dir": str(token_dir),
        "token_format": _token_dir_format(token_dir),
        "pairs_jsonl": str(_pairs_jsonl_path(token_dir)),
        "num_images": len(unique_image_ids),
        "num_texts": len(row_image_ids),
        "dim_img": dim_img,
        "dim_txt": dim_txt,
        "batch_size": batch_size,
        "device": str(resolved_device),
        "parameter_names": pal_trainable_parameter_names(model),
        "metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _read_pairs_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pairs_jsonl_path(token_dir: Path) -> Path:
    metadata_path = token_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pairs = metadata.get("pairs_jsonl")
        if pairs:
            candidate = Path(pairs)
            if candidate.is_absolute():
                return candidate
            if (token_dir / candidate).exists():
                return token_dir / candidate
            return candidate
    return token_dir / "pairs.jsonl"


@torch.no_grad()
def _encode_token_rows(
    model: ProjectionFreeAnchorLearning,
    token_dir: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if _token_dir_format(token_dir) == "chunks":
        return _encode_chunked_rows(model, token_dir, batch_size, device)
    tensors = load_token_tensors(token_dir, map_location="cpu")
    image_features = encode_images_batched(model, tensors.image_tokens, batch_size=batch_size, device=device)
    text_features = encode_texts_batched(model, tensors.text_tokens, tensors.text_mask, batch_size=batch_size, device=device)
    return image_features, text_features, tensors.dim_img, tensors.dim_txt


@torch.no_grad()
def _encode_chunked_rows(
    model: ProjectionFreeAnchorLearning,
    token_dir: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    metadata = json.loads((token_dir / "metadata.json").read_text(encoding="utf-8"))
    chunks = sorted(metadata["chunks"], key=lambda item: int(item["chunk_index"]))
    image_features: list[torch.Tensor] = []
    text_features: list[torch.Tensor] = []
    dim_img = 0
    dim_txt = 0
    for chunk in chunks:
        image_tokens = torch.load(token_dir / chunk["image_tokens"], map_location="cpu", weights_only=True)
        text_tokens = torch.load(token_dir / chunk["text_tokens"], map_location="cpu", weights_only=True)
        text_mask = torch.load(token_dir / chunk["text_mask"], map_location="cpu", weights_only=True)
        dim_img = int(image_tokens.shape[-1])
        dim_txt = int(text_tokens.shape[-1])
        image_features.append(
            encode_images_batched(model, image_tokens, batch_size=batch_size, device=device)
        )
        text_features.append(
            encode_texts_batched(model, text_tokens, text_mask, batch_size=batch_size, device=device)
        )
    if not image_features:
        raise ValueError(f"no chunks found in {token_dir}")
    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0), dim_img, dim_txt


def _token_dir_format(token_dir: Path) -> str:
    metadata_path = token_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("format") == "chunks":
                return "chunks"
            return str(metadata.get("format", "monolithic"))
        except json.JSONDecodeError:
            return "monolithic"
    return "monolithic"


@torch.no_grad()
def _evaluate_retrieval_chunked(
    model: ProjectionFreeAnchorLearning,
    token_dir: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    metadata = json.loads((token_dir / "metadata.json").read_text(encoding="utf-8"))
    chunks = sorted(metadata["chunks"], key=lambda item: int(item["chunk_index"]))
    image_features: list[torch.Tensor] = []
    text_features: list[torch.Tensor] = []
    num_samples = 0
    dim_img = None
    dim_txt = None
    for chunk in chunks:
        image_tokens = torch.load(token_dir / chunk["image_tokens"], map_location="cpu", weights_only=True)
        text_tokens = torch.load(token_dir / chunk["text_tokens"], map_location="cpu", weights_only=True)
        text_mask = torch.load(token_dir / chunk["text_mask"], map_location="cpu", weights_only=True)
        if image_tokens.shape[0] != text_tokens.shape[0]:
            raise ValueError(f"chunk {chunk['chunk_index']} image/text sample count mismatch")
        dim_img = int(image_tokens.shape[-1])
        dim_txt = int(text_tokens.shape[-1])
        num_samples += int(image_tokens.shape[0])
        image_features.append(
            encode_images_batched(model, image_tokens, batch_size=batch_size, device=device)
        )
        text_features.append(
            encode_texts_batched(model, text_tokens, text_mask, batch_size=batch_size, device=device)
        )
    if not image_features:
        raise ValueError(f"no chunks found in {token_dir}")
    image_all = torch.cat(image_features, dim=0)
    text_all = torch.cat(text_features, dim=0)
    metrics = retrieval_metrics(image_all, text_all, percent=True)
    metrics["mean_recall"] = sum(metrics.values()) / len(metrics)
    return {
        "token_format": "chunks",
        "num_samples": num_samples,
        "dim_img": dim_img,
        "dim_txt": dim_txt,
        "metrics": metrics,
    }


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    retrieval = sub.add_parser("retrieval", help="Evaluate one-to-one image/text retrieval on cached tokens.")
    retrieval.add_argument("--checkpoint", type=Path, required=True)
    retrieval.add_argument("--token-dir", type=Path, required=True)
    retrieval.add_argument("--output", type=Path, required=True)
    retrieval.add_argument("--batch-size", type=int, default=256)
    retrieval.add_argument("--device", default="auto")
    multicaption = sub.add_parser("retrieval-multicaption", help="Evaluate retrieval with multiple captions per image using pairs.jsonl image_id positives.")
    multicaption.add_argument("--checkpoint", type=Path, required=True)
    multicaption.add_argument("--token-dir", type=Path, required=True)
    multicaption.add_argument("--output", type=Path, required=True)
    multicaption.add_argument("--batch-size", type=int, default=256)
    multicaption.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "retrieval":
        result = evaluate_retrieval_token_dir(
            checkpoint_path=args.checkpoint,
            token_dir=args.token_dir,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "retrieval-multicaption":
        result = evaluate_multicaption_retrieval_token_dir(
            checkpoint_path=args.checkpoint,
            token_dir=args.token_dir,
            output_path=args.output,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
