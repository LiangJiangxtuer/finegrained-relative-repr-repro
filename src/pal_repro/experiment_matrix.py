"""Paper reproduction matrix and target metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_experiment_matrix(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the YAML reproduction matrix."""

    matrix = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(matrix, dict):
        raise ValueError("reproduction matrix must be a mapping.")
    for key in ["paper", "experiments", "ablations"]:
        if key not in matrix:
            raise ValueError(f"reproduction matrix missing {key!r}.")
    if "main_pal_k512" not in matrix["experiments"]:
        raise ValueError("matrix must contain experiments.main_pal_k512.")
    return matrix


def paper_target_metrics() -> dict[str, Any]:
    """Return paper-reported PAL target metrics for the main experiment."""

    return {
        "classification_top1": {
            "STL10": 95.3,
            "CIFAR100": 48.8,
            "Caltech101": 60.9,
            "DTD": 17.7,
            "EuroSAT": 34.6,
        },
        "retrieval_r1": {
            "Flickr30k": {"i2t": 76.3, "t2i": 61.8},
            "COCO": {"i2t": 56.3, "t2i": 42.6},
        },
        "segmentation_miou_fg": {
            "VOC20": 32.3,
            "Context": 25.5,
            "ADE20K": 13.8,
        },
        "ablation_token_usage_cap": {
            "global_only": {"avg_cls": 48.4, "avg_ret": 43.9, "avg_seg": 7.3},
            "full_tokens_mean": {"avg_cls": 49.3, "avg_ret": 48.4, "avg_seg": 16.3},
            "full_tokens_cap": {"avg_cls": 51.5, "avg_ret": 59.3, "avg_seg": 23.9},
        },
        "ablation_pool_temperature": {
            0.02: {"avg_cls": 51.1, "avg_ret": 58.8, "avg_seg": 21.6},
            0.03: {"avg_cls": 51.5, "avg_ret": 59.3, "avg_seg": 23.9},
            0.05: {"avg_cls": 50.4, "avg_ret": 57.9, "avg_seg": 23.2},
            0.07: {"avg_cls": 50.2, "avg_ret": 55.5, "avg_seg": 21.0},
            0.10: {"avg_cls": 49.7, "avg_ret": 52.9, "avg_seg": 18.5},
        },
    }
