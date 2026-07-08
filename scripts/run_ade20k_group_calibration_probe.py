#!/usr/bin/env python3
"""Targeted ADE20K class/group calibration probes.

This runner keeps the recovered ADE20K dense-token protocol fixed
(DINOv2/RoBERTa last_hidden_state, processor target, clean aliases,
--ignore-zero, tau_p=0.07 checkpoint) and applies small explicit logit
biases to selected class groups. It is diagnostic: rows use ADE20K val labels
for analysis and must be full-confirmed before any paper-grade promotion.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FREQUENT_CLASSES = ("wall", "building", "sky", "floor", "tree", "person", "road")
TEMPLATES = (
    "a photo of {class_name}",
    "a cropped photo of {class_name}",
    "a close-up photo of {class_name}",
    "a clean photo of {class_name}",
)


@dataclass(frozen=True)
class CalibrationCase:
    name: str
    class_bias: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


def default_cases() -> list[CalibrationCase]:
    frequent = "wall,building,sky,floor,tree,person,road"
    underpred = "wall,sky,person,road"
    stuff = "wall,sky,floor,tree"
    spurious = "house,screen door,bookcase,skyscraper,swivel chair"
    return [
        CalibrationCase("baseline_no_bias", description="Recovered last_hidden_state protocol without manual class bias."),
        CalibrationCase("boost_frequent7_p001", (f"{frequent}=0.01",)),
        CalibrationCase("boost_frequent7_p002", (f"{frequent}=0.02",)),
        CalibrationCase("boost_frequent7_p004", (f"{frequent}=0.04",)),
        CalibrationCase("boost_underpred4_p001", (f"{underpred}=0.01",)),
        CalibrationCase("boost_underpred4_p002", (f"{underpred}=0.02",)),
        CalibrationCase("boost_underpred4_p004", (f"{underpred}=0.04",)),
        CalibrationCase("boost_stuff4_p002", (f"{stuff}=0.02",)),
        CalibrationCase("boost_wall_sky_p002", ("wall,sky=0.02",)),
        CalibrationCase("boost_wall_sky_p004", ("wall,sky=0.04",)),
        CalibrationCase("suppress_spurious5_m001", (f"{spurious}=-0.01",)),
        CalibrationCase("suppress_spurious5_m002", (f"{spurious}=-0.02",)),
        CalibrationCase("underpred_p002_spurious_m001", (f"{underpred}=0.02", f"{spurious}=-0.01")),
        CalibrationCase("underpred_p002_spurious_m002", (f"{underpred}=0.02", f"{spurious}=-0.02")),
        CalibrationCase("wall_sky_p002_spurious_m001", ("wall,sky=0.02", f"{spurious}=-0.01")),
    ]


def build_command(args: argparse.Namespace, case: CalibrationCase, output: Path) -> list[str]:
    cmd = [
        str(args.python),
        str(args.root / "scripts/evaluate_segmentation.py"),
        "--dataset", "ADE20K",
        "--checkpoint", str(args.checkpoint),
        "--output", str(output),
        "--batch-size", str(args.batch_size),
        "--target-frame", "processor",
        "--alias-policy", "clean",
        "--ignore-zero",
        "--limit", str(args.limit),
        "--device", args.device,
        "--local-files-only",
    ]
    for template in TEMPLATES:
        cmd.extend(["--prompt-template", template])
    for spec in case.class_bias:
        cmd.extend(["--class-bias", spec])
    return cmd


def summarize_output(case: CalibrationCase, output: Path) -> dict[str, Any]:
    data = json.loads(output.read_text(encoding="utf-8"))
    class_counts = data.get("class_counts", {})
    frequent = {name: class_counts.get(name, {}) for name in FREQUENT_CLASSES}
    return {
        "name": case.name,
        "description": case.description,
        "class_bias": list(case.class_bias),
        "output": str(output),
        "miou": float(data["metrics"]["foreground_miou"]),
        "top_pred": data.get("top_predicted_classes", [])[:8],
        "frequent": frequent,
    }


def run_case(args: argparse.Namespace, case: CalibrationCase) -> dict[str, Any]:
    output = args.output_dir / f"{case.name}.json"
    log = args.output_dir / f"{case.name}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and output.exists():
        return summarize_output(case, output)
    cmd = build_command(args, case, output)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.root / "src")
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=args.root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"case {case.name} failed with exit {proc.returncode}; see {log}")
    return summarize_output(case, output)


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda item: item["miou"], reverse=True)
    baseline = next((row for row in rows if row["name"] == "baseline_no_bias"), None)
    baseline_miou = baseline["miou"] if baseline else None
    for row in rows:
        row["delta_vs_baseline"] = None if baseline_miou is None else row["miou"] - baseline_miou
    payload = {
        "mode": "ade20k_targeted_group_calibration_probe",
        "protocol": "last_hidden_state_dense_tokens_clean_alias_ignore_zero_limit_probe",
        "limit": args.limit,
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "rows": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {summary_path}")
    print("name\tmiou\tdelta\tclass_bias")
    for row in rows:
        delta = row["delta_vs_baseline"]
        print(f"{row['name']}\t{row['miou']:.4f}\t{delta if delta is not None else 'n/a'}\t{row['class_bias']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path("/home/hnxxzy/miniconda3/envs/ovvs/bin/python"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ablations/tau_0_07/checkpoint.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/ade20k_group_calibration_limit64"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.root = args.root.resolve()
    args.output_dir = (args.root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.checkpoint = (args.root / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    cases = default_cases()
    if args.list or not args.run:
        print(json.dumps([case.__dict__ for case in cases], indent=2, sort_keys=True))
    if args.run:
        rows = [run_case(args, case) for case in cases]
        write_summary(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
