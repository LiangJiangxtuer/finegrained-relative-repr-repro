#!/usr/bin/env python3
"""ADE20K dense-token/layer/calibration recovery probes.

This runner keeps the best current ADE20K protocol fixed (processor target,
clean aliases, four-template prompt ensemble, --ignore-zero, tau_p=0.07
checkpoint) and changes one dense-token or calibration factor at a time.
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
class ProbeCase:
    name: str
    vision_layer: int | None = None
    text_layer: int | None = None
    vision_layer_ensemble: tuple[int, ...] = field(default_factory=tuple)
    text_layer_ensemble: tuple[int, ...] = field(default_factory=tuple)
    logit_calibration: str = "none"
    image_size: int | None = None
    description: str = ""


def default_cases() -> list[ProbeCase]:
    return [
        ProbeCase("current_hidden_v-1_t-2", vision_layer=-1, text_layer=-2, description="Current selected hidden-state protocol."),
        ProbeCase("last_hidden_state_vlast_tlast", description="Use model last_hidden_state for both encoders instead of hidden_states[-k]."),
        ProbeCase("last_vision_hidden_text_t-2", text_layer=-2, description="DINOv2 last_hidden_state plus RoBERTa hidden_states[-2]."),
        ProbeCase("hidden_v-1_last_text", vision_layer=-1, description="DINOv2 hidden_states[-1] plus RoBERTa last_hidden_state."),
        ProbeCase("hidden_v-1_t-1", vision_layer=-1, text_layer=-1),
        ProbeCase("hidden_v-1_t-4", vision_layer=-1, text_layer=-4),
        ProbeCase("hidden_v-1_t-6", vision_layer=-1, text_layer=-6),
        ProbeCase("hidden_v-2_t-2", vision_layer=-2, text_layer=-2),
        ProbeCase("hidden_v-2_t-4", vision_layer=-2, text_layer=-4),
        ProbeCase("vision_ens_v-1_v-2_t-2", vision_layer_ensemble=(-1, -2), text_layer=-2),
        ProbeCase("vision_ens_v-1_v-2_v-4_t-2", vision_layer_ensemble=(-1, -2, -4), text_layer=-2),
        ProbeCase("text_ens_v-1_t-2_t-4", vision_layer=-1, text_layer_ensemble=(-2, -4)),
        ProbeCase("both_ens_v-1_v-2_t-2_t-4", vision_layer_ensemble=(-1, -2), text_layer_ensemble=(-2, -4)),
        ProbeCase("cal_center_v-1_t-2", vision_layer=-1, text_layer=-2, logit_calibration="image-class-center"),
        ProbeCase("cal_zscore_v-1_t-2", vision_layer=-1, text_layer=-2, logit_calibration="image-class-zscore"),
        ProbeCase("clean_size336_v-1_t-2", vision_layer=-1, text_layer=-2, image_size=336),
    ]


def build_command(args: argparse.Namespace, case: ProbeCase, output: Path) -> list[str]:
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
    if case.vision_layer is not None:
        cmd.extend(["--vision-layer", str(case.vision_layer)])
    if case.text_layer is not None:
        cmd.extend(["--text-layer", str(case.text_layer)])
    for layer in case.vision_layer_ensemble:
        cmd.extend(["--vision-layer-ensemble", str(layer)])
    for layer in case.text_layer_ensemble:
        cmd.extend(["--text-layer-ensemble", str(layer)])
    if case.logit_calibration != "none":
        cmd.extend(["--logit-calibration", case.logit_calibration])
    if case.image_size is not None:
        cmd.extend(["--image-size", str(case.image_size)])
    return cmd


def summarize_output(case: ProbeCase, output: Path) -> dict[str, Any]:
    data = json.loads(output.read_text(encoding="utf-8"))
    class_counts = data.get("class_counts", {})
    frequent = {
        name: class_counts.get(name, {"iou": data.get("class_ious", {}).get(name)})
        for name in FREQUENT_CLASSES
    }
    return {
        "name": case.name,
        "description": case.description,
        "output": str(output),
        "miou": float(data["metrics"]["foreground_miou"]),
        "vision_layer": data.get("vision_layer"),
        "text_layer": data.get("text_layer"),
        "vision_layer_ensemble": data.get("vision_layer_ensemble"),
        "text_layer_ensemble": data.get("text_layer_ensemble"),
        "logit_calibration": data.get("logit_calibration"),
        "image_size": data.get("image_size"),
        "top_pred": data.get("top_predicted_classes", [])[:5],
        "frequent": frequent,
    }


def run_case(args: argparse.Namespace, case: ProbeCase) -> dict[str, Any]:
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
    payload = {
        "mode": "ade20k_dense_token_layer_calibration_recovery_probe",
        "limit": args.limit,
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "rows": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WROTE {summary_path}")
    print("name\tmiou\tvision\ttext\tcalibration\timage_size")
    for row in rows:
        print(
            f"{row['name']}\t{row['miou']:.4f}\t"
            f"{row.get('vision_layer') or row.get('vision_layer_ensemble')}\t"
            f"{row.get('text_layer') or row.get('text_layer_ensemble')}\t"
            f"{row.get('logit_calibration')}\t{row.get('image_size')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path("/home/hnxxzy/miniconda3/envs/ovvs/bin/python"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/ablations/tau_0_07/checkpoint.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/ade20k_dense_protocol_recovery_limit64"))
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
