#!/usr/bin/env python3
"""Run zero-shot classification prompt-template sweeps for PAL."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_DATASETS = ["STL10", "CIFAR100", "Caltech101", "DTD", "EuroSAT"]


def safe_name(value: str) -> str:
    value = value.replace("{class_name}", "class")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()[:80] or "template"


def run_prompt_sweep(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for template in args.template:
        for dataset in args.dataset:
            output = args.output_dir / safe_name(template) / f"{dataset.lower()}_classification.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "evaluate_classification.py"),
                "--dataset", dataset,
                "--checkpoint", str(args.checkpoint),
                "--output", str(output),
                "--batch-size", str(args.batch_size),
                "--prompt-template", template,
            ]
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            if args.local_files_only:
                cmd.append("--local-files-only")
            print("RUN", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            payload = json.loads(output.read_text(encoding="utf-8"))
            row = {
                "dataset": dataset,
                "template": template,
                "output": str(output),
                "top1": payload["metrics"]["top1"],
                "top5": payload["metrics"].get("top5"),
                "num_samples": payload.get("num_samples"),
            }
            rows.append(row)
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in args.dataset:
        subset = [row for row in rows if row["dataset"] == dataset]
        if subset:
            by_dataset[dataset] = max(subset, key=lambda row: row["top1"])
    summary = {
        "checkpoint": str(args.checkpoint),
        "templates": args.template,
        "datasets": args.dataset,
        "rows": rows,
        "best_by_dataset": by_dataset,
        "average_best_top1": sum(row["top1"] for row in by_dataset.values()) / max(len(by_dataset), 1),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", choices=DEFAULT_DATASETS, default=[])
    parser.add_argument("--template", action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.dataset:
        args.dataset = DEFAULT_DATASETS
    run_prompt_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
