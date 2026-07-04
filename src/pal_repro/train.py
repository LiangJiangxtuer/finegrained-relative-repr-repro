"""Training entry point for strict PAL reproduction."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pal_repro.data import TokenTensorDataset, load_token_tensors, split_indices
from pal_repro.eval import retrieval_metrics
from pal_repro.losses import symmetric_info_nce_loss
from pal_repro.models.pal import ProjectionFreeAnchorLearning, pal_trainable_parameter_names


@dataclass
class TrainConfig:
    """Configuration for PAL training on pre-extracted token tensors."""

    data_dir: Path
    output_dir: Path
    num_anchors: int = 512
    pool_temperature: float = 0.03
    pooling_mode: str = "cap"
    contrastive_temperature: float = 0.07
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    train_size: int | None = None
    train_fraction: float = 0.8
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.pooling_mode not in {"cap", "mean", "global"}:
            raise ValueError("pooling_mode must be one of: cap, mean, global.")


def seed_everything(seed: int) -> None:
    """Set RNG seeds used by the lightweight reproduction pipeline."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve configured device string."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return requested


def _batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image, text, mask = batch
    return (
        image.to(device=device, dtype=torch.float32),
        text.to(device=device, dtype=torch.float32),
        mask.to(device=device),
    )


@torch.no_grad()
def _encode_dataset(
    model: ProjectionFreeAnchorLearning,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    image_features: list[torch.Tensor] = []
    text_features: list[torch.Tensor] = []
    for batch in loader:
        image, text, mask = _batch_to_device(batch, device)
        output = model(image, text, mask)
        image_features.append(output.image.detach().cpu())
        text_features.append(output.text.detach().cpu())
    return torch.cat(image_features, dim=0), torch.cat(text_features, dim=0)


def _maybe_eval(
    model: ProjectionFreeAnchorLearning,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    if len(loader.dataset) == 0:
        return {}
    image_eval, text_eval = _encode_dataset(model, loader, device)
    return retrieval_metrics(image_eval, text_eval)


def train_pal(config: TrainConfig) -> dict[str, Any]:
    """Train strict PAL on token tensors and write metrics/checkpoint."""

    seed_everything(config.seed)
    device = resolve_device(config.device)
    tensors = load_token_tensors(config.data_dir, map_location="cpu")
    train_idx, eval_idx = split_indices(
        tensors.num_samples,
        train_size=config.train_size,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )

    train_dataset = TokenTensorDataset(tensors, train_idx)
    eval_dataset = TokenTensorDataset(tensors, eval_idx)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = ProjectionFreeAnchorLearning(
        dim_img=tensors.dim_img,
        dim_txt=tensors.dim_txt,
        num_anchors=config.num_anchors,
        pool_temperature=config.pool_temperature,
        pooling_mode=config.pooling_mode,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            image, text, mask = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(image, text, mask)
            loss = symmetric_info_nce_loss(
                output.image,
                output.text,
                temperature=config.contrastive_temperature,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        eval_metrics = _maybe_eval(model, eval_loader, device)
        row = {"epoch": float(epoch), "train_loss": mean_loss, **eval_metrics}
        history.append(row)

    final_eval = _maybe_eval(model, eval_loader, device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": _jsonable_config(config),
        "dim_img": tensors.dim_img,
        "dim_txt": tensors.dim_txt,
        "parameter_names": pal_trainable_parameter_names(model),
    }
    torch.save(checkpoint, config.output_dir / "checkpoint.pt")

    result: dict[str, Any] = {
        "config": _jsonable_config(config),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "device": str(device),
        "num_samples": tensors.num_samples,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "dim_img": tensors.dim_img,
        "dim_txt": tensors.dim_txt,
        "parameter_names": pal_trainable_parameter_names(model),
        "history": history,
        "eval": final_eval,
        "final_train_loss": history[-1]["train_loss"],
        "checkpoint": str(config.output_dir / "checkpoint.pt"),
    }
    (config.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _jsonable_config(config: TrainConfig) -> dict[str, Any]:
    raw = asdict(config)
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in raw.items()}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    model = raw.get("model", {})
    training = raw.get("training", {})
    data = raw.get("data", {})
    return {
        "data_dir": data.get("token_dir"),
        "output_dir": training.get("output_dir", "outputs/pal_run"),
        "num_anchors": model.get("num_anchors", 512),
        "pool_temperature": model.get("pool_temperature", 0.03),
        "pooling_mode": model.get("pooling_mode", "cap"),
        "contrastive_temperature": training.get("contrastive_temperature", 0.07),
        "epochs": training.get("epochs", 20),
        "batch_size": training.get("batch_size", 256),
        "lr": training.get("lr", 1.0e-3),
        "weight_decay": training.get("weight_decay", 1.0e-4),
        "train_size": data.get("train_size"),
        "train_fraction": data.get("train_fraction", 0.8),
        "seed": training.get("seed", 42),
        "device": training.get("device", "auto"),
        "num_workers": training.get("num_workers", 0),
    }


def config_from_yaml(path: str | Path, preset: str | None = None) -> TrainConfig:
    """Load TrainConfig from a YAML file with optional preset override."""

    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Empty config: {path}")
    selected = {key: value for key, value in raw.items() if key != "presets"}
    if preset is not None:
        presets = raw.get("presets", {})
        if preset not in presets:
            raise KeyError(f"Unknown preset {preset!r}; available: {sorted(presets)}")
        selected = _deep_update(selected, presets[preset].copy())
    flat = _flatten_config(selected)
    if flat["data_dir"] is None:
        raise ValueError("Config must define data.token_dir.")
    return TrainConfig(**flat)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train strict PAL on token tensors.")
    parser.add_argument("--config", type=Path, default=Path("configs/pal_strict.yaml"))
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-anchors", type=int, default=None)
    parser.add_argument("--pool-temperature", type=float, default=None)
    parser.add_argument("--pooling-mode", choices=["cap", "mean", "global"], default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_yaml(args.config, preset=args.preset)
    for attr, field in [
        ("data_dir", "data_dir"),
        ("output_dir", "output_dir"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("num_anchors", "num_anchors"),
        ("pool_temperature", "pool_temperature"),
        ("pooling_mode", "pooling_mode"),
        ("device", "device"),
        ("lr", "lr"),
        ("train_size", "train_size"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            setattr(cfg, field, value)
    cfg.__post_init__()
    result = train_pal(cfg)
    print(json.dumps({"metrics": result["eval"], "checkpoint": result["checkpoint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
