#!/usr/bin/env python3
"""Run zero-shot classification evaluation for trained PAL ablation checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEMPLATES = [
    "a photo of {class_name}",
    "a cropped photo of {class_name}",
    "a close-up photo of {class_name}",
    "a clean photo of {class_name}",
]
PAPER_AVG_TOP1 = 51.46


@dataclass(frozen=True)
class Variant:
    group: str
    label: str
    checkpoint: Path

    @property
    def name(self) -> str:
        return f"{self.group}_{self.label}"


def build_variants(root: Path) -> list[Variant]:
    return [
        *(Variant("k", str(k), root / f"outputs/ablations/k_{k}/checkpoint.pt") for k in (32, 64, 128, 256, 512)),
        Variant("tau", "0_02", root / "outputs/ablations/tau_0_02/checkpoint.pt"),
        Variant("tau", "0_03", root / "outputs/pal_k512_coco2014_full/checkpoint.pt"),
        Variant("tau", "0_05", root / "outputs/ablations/tau_0_05/checkpoint.pt"),
        Variant("tau", "0_07", root / "outputs/ablations/tau_0_07/checkpoint.pt"),
        Variant("tau", "0_10", root / "outputs/ablations/tau_0_10/checkpoint.pt"),
        Variant("token_usage", "global", root / "outputs/ablations/token_usage_global/checkpoint.pt"),
        Variant("token_usage", "mean", root / "outputs/ablations/token_usage_mean/checkpoint.pt"),
        Variant("token_usage", "cap", root / "outputs/ablations/token_usage_cap/checkpoint.pt"),
    ]


def variant_output_dir(root: Path, output_dir: Path, variant: Variant) -> Path:
    return root / output_dir / variant.group / variant.label


def run_variant(args: argparse.Namespace, variant: Variant) -> dict[str, Any]:
    out_dir = variant_output_dir(args.root, args.output_dir, variant)
    summary_path = out_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        print(f"SKIP existing {variant.name}: {summary_path}", flush=True)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return {"variant": variant.name, "status": "skipped", **summarize_payload(variant, summary_path, payload)}
    if not variant.checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint for {variant.name}: {variant.checkpoint}")
    cmd = [
        sys.executable,
        str(args.root / "scripts/run_prompt_sweep.py"),
        "--checkpoint", str(variant.checkpoint),
        "--output-dir", str(out_dir),
        "--batch-size", str(args.batch_size),
        "--ensemble",
    ]
    for template in TEMPLATES:
        cmd.extend(["--template", template])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    log_path = out_dir / "run.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"RUN {variant.name}: {' '.join(cmd)}", flush=True)
    print(f"LOG {variant.name}: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=args.root, env=pythonpath_env(args.root), stdout=log, stderr=subprocess.STDOUT, check=True)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"variant": variant.name, "status": "completed", **summarize_payload(variant, summary_path, payload)}


def summarize_payload(variant: Variant, summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    average_top1 = float(payload["average_top1"])
    return {
        "group": variant.group,
        "label": variant.label,
        "checkpoint": str(variant.checkpoint),
        "summary": str(summary_path),
        "average_top1": average_top1,
        "paper_average_top1": PAPER_AVG_TOP1,
        "gap": average_top1 - PAPER_AVG_TOP1,
        "relative_percent": average_top1 / PAPER_AVG_TOP1 * 100.0,
        "rows": payload.get("rows", []),
    }


def write_combined_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["group"]), []).append(row)
    payload = {
        "mode": "fixed_prompt_ensemble_classification_ablation",
        "prompt_templates": TEMPLATES,
        "paper_average_top1": PAPER_AVG_TOP1,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "local_files_only": args.local_files_only,
        "groups": by_group,
        "rows": rows,
        "protocol": {
            "source_paper_claim": "PAL ablation tables report average zero-shot classification top-1.",
            "dataset_split": "torchvision/default test split per dataset loader",
            "prompt_policy": {"mode": "fixed_ensemble", "templates": TEMPLATES},
            "known_deviation": "Classification ablation uses the repository's fixed fair prompt ensemble; paper exact prompt set is not fully specified.",
        },
    }
    output_path = args.root / args.output_dir / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {output_path}", flush=True)


def pythonpath_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root / "src") + ((":" + prior) if prior else "")
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablations/classification"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", action="append", default=[], help="Variant name such as k_32, tau_0_02, or token_usage_cap.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.root = args.root.resolve()
    variants = build_variants(args.root)
    if args.only:
        wanted = set(args.only)
        variants = [variant for variant in variants if variant.name in wanted]
        missing = wanted - {variant.name for variant in variants}
        if missing:
            raise KeyError(f"unknown variants: {sorted(missing)}")
    if args.list or not args.run:
        print(json.dumps([
            {
                "name": variant.name,
                "group": variant.group,
                "label": variant.label,
                "checkpoint": str(variant.checkpoint),
                "output_dir": str(variant_output_dir(args.root, args.output_dir, variant)),
            }
            for variant in variants
        ], indent=2, sort_keys=True))
    if args.run:
        rows = [run_variant(args, variant) for variant in variants]
        write_combined_summary(args, rows)
        print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
