#!/usr/bin/env python3
"""Summarize ADE20K frequent-class segmentation errors and calibration probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FREQUENT_CLASSES = ("wall", "building", "sky", "floor", "tree", "person", "road")
KEY_PROBE_ROWS = (
    "alias_first",
    "alias_all",
    "alias_clean",
    "prior_ratio_a0_1",
    "prior_ratio_a0_25",
    "prior_train_a0_25",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}%"


def row_by_name(summary: dict[str, Any], name: str) -> dict[str, Any] | None:
    for row in summary.get("rows", []):
        if row.get("config", {}).get("name") == name:
            return row
    return None


def frequent_metrics(row: dict[str, Any], class_name: str) -> dict[str, float]:
    metrics = row["frequent"][class_name]
    pred = float(metrics.get("pred_pixels", 0.0))
    target = float(metrics.get("target_pixels", 0.0))
    inter = float(metrics.get("intersection", 0.0))
    return {
        "iou": float(metrics.get("iou", 0.0)),
        "pred_target": pred / target if target else 0.0,
        "recall": inter / target if target else 0.0,
        "precision": inter / pred if pred else 0.0,
    }


def full_frequent_ious(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = load_json(path)
    class_ious = data.get("class_ious", {})
    return {name: class_ious.get(name) for name in FREQUENT_CLASSES}


def clean_full_rows(summary_path: Path) -> list[dict[str, Any]]:
    if not summary_path.exists():
        return []
    data = load_json(summary_path)
    rows: list[dict[str, Any]] = []
    for row in data.get("rows", []):
        variant_rows = row.get("rows", [])
        ade_row = next((item for item in variant_rows if item.get("dataset") == "ADE20K"), None)
        if ade_row is None:
            continue
        rows.append({
            "variant": row.get("variant"),
            "group": row.get("group"),
            "label": row.get("label"),
            "miou": ade_row.get("foreground_miou"),
            "output": ade_row.get("output"),
        })
    return rows


def write_report(args: argparse.Namespace) -> None:
    summary = load_json(args.probe_summary)
    lines: list[str] = [
        "# ADE20K Frequent-Class Error Analysis",
        "",
        "This report is generated from real evaluator JSON outputs. The frequent classes are `wall/building/sky/floor/tree/person/road`.",
        "",
        "## Sources",
        "",
        f"- 64-sample alias/prior probe summary: `{args.probe_summary}`",
        f"- Selected full ADE20K result: `{args.full_selected}`",
        f"- Clean-alias full ADE20K all-variant summary: `{args.clean_full_summary}`" + ("" if args.clean_full_summary.exists() else " (not available yet)"),
        f"- Dense-token/layer recovery summary: `{args.dense_recovery_summary}`" + ("" if args.dense_recovery_summary.exists() else " (not available yet)"),
        "",
    ]

    lines.extend([
        "## 64-sample alias / prior probe overview",
        "",
        "| Probe | Alias policy | Prior | alpha | mIoU | Top predicted class |",
        "|---|---|---|---:|---:|---|",
    ])
    for row in sorted(summary.get("rows", []), key=lambda item: item.get("miou", 0.0), reverse=True):
        cfg = row.get("config", {})
        top = row.get("top_pred", [{}])[0].get("class_name", "n/a")
        lines.append(
            f"| `{cfg.get('name')}` | {cfg.get('alias')} | {cfg.get('prior')} | {cfg.get('alpha')} | {fmt(row.get('miou'), 3)} | {top} |"
        )
    lines.append("")

    lines.extend([
        "## Frequent-class detail on the 64-sample probe",
        "",
        "Values are IoU / pred-target ratio / recall / precision. A low pred-target ratio means the class is rarely predicted; a high ratio with low precision means the class is overused.",
        "",
    ])
    for name in KEY_PROBE_ROWS:
        row = row_by_name(summary, name)
        if row is None:
            continue
        cfg = row.get("config", {})
        lines.extend([
            f"### `{name}` (alias={cfg.get('alias')}, prior={cfg.get('prior')}, alpha={cfg.get('alpha')}, mIoU={fmt(row.get('miou'), 3)})",
            "",
            "| Class | IoU | pred/target | recall | precision |",
            "|---|---:|---:|---:|---:|",
        ])
        for class_name in FREQUENT_CLASSES:
            metrics = frequent_metrics(row, class_name)
            lines.append(
                f"| {class_name} | {fmt(metrics['iou'])} | {fmt(metrics['pred_target'], 3)} | {pct(100 * metrics['recall'])} | {pct(100 * metrics['precision'])} |"
            )
        lines.append("")

    full_ious = full_frequent_ious(args.full_selected)
    if full_ious is not None:
        data = load_json(args.full_selected)
        lines.extend([
            "## Selected full ADE20K frequent-class IoU",
            "",
            f"Selected full result mIoU: `{fmt(data.get('metrics', {}).get('foreground_miou'), 3)}` with alias policy `{data.get('alias_policy')}`, ignore index `{data.get('ignore_index')}`.",
            "",
            "| Class | Full IoU |",
            "|---|---:|",
        ])
        for class_name in FREQUENT_CLASSES:
            lines.append(f"| {class_name} | {fmt(full_ious.get(class_name))} |")
        lines.append("")

    clean_rows = clean_full_rows(args.clean_full_summary)
    if clean_rows:
        lines.extend([
            "## Full ADE20K clean-alias rows",
            "",
            "These rows use `--alias-policy clean --ignore-zero` and share frozen image forward across ablation checkpoints.",
            "",
            "| Variant | Group | Label | ADE20K mIoU | Output |",
            "|---|---|---|---:|---|",
        ])
        for row in sorted(clean_rows, key=lambda item: item.get("miou", 0.0), reverse=True):
            lines.append(
                f"| `{row.get('variant')}` | {row.get('group')} | {row.get('label')} | {fmt(row.get('miou'), 3)} | `{row.get('output')}` |"
            )
        lines.append("")

    if args.dense_recovery_summary.exists():
        dense_summary = load_json(args.dense_recovery_summary)
        rows = dense_summary.get("rows", [])
        if rows:
            lines.extend([
                "## Dense-token / layer recovery full confirmation",
                "",
                "These rows keep clean ADE20K aliases and `--ignore-zero` fixed, then vary dense-token layer selection on the full ADE20K validation split.",
                "",
                "| Probe | ADE20K mIoU | Output |",
                "|---|---:|---|",
            ])
            for row in sorted(rows, key=lambda item: item.get("miou", 0.0), reverse=True):
                lines.append(f"| `{row.get('name')}` | {fmt(row.get('miou'), 3)} | `{row.get('output')}` |")
            lines.append("")

    best_alias = row_by_name(summary, "alias_first")
    prior_a01 = row_by_name(summary, "prior_ratio_a0_1")
    prior_a025 = row_by_name(summary, "prior_ratio_a0_25")
    lines.extend([
        "## Calibration / prior-correction conclusion",
        "",
        "- The uncalibrated alias probes remain best on overall mIoU. `alias_first`, `alias_all`, and `alias_clean` are tightly clustered around 3.0 mIoU on the 64-sample probe.",
        "- Class-prior logit biases improve some frequent-class recall (especially wall/sky/floor) but collapse the long-tail mean mIoU: the best tested prior row is below the uncalibrated alias rows.",
        "- The prior bias is too blunt: `ade20k-ratio` with alpha >= 0.25 makes wall dominate predictions, while `ade20k-train-count` similarly overweights frequent classes and suppresses person/road.",
        "- A foreground/background calibration is not indicated for ADE20K beyond `--ignore-zero`: the evaluation has 150 labeled foreground classes and label id 0 should be void/ignored, not a learned background class. For VOC/Context, explicit background calibration could be studied separately, but it would be a protocol change rather than an ADE20K class-prior fix.",
    ])
    if best_alias and prior_a01 and prior_a025:
        lines.append(
            f"- Concrete probe numbers: uncalibrated `alias_first` mIoU `{fmt(best_alias.get('miou'), 3)}` vs `ade20k-ratio` alpha=0.1 `{fmt(prior_a01.get('miou'), 3)}` and alpha=0.25 `{fmt(prior_a025.get('miou'), 3)}`."
        )
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-summary", type=Path, default=Path("outputs/diagnostics/ade20k_dense_debug/alias_prior_limit64_ignore0/summary.json"))
    parser.add_argument("--full-selected", type=Path, default=Path("outputs/diagnostics/ade20k_dense_protocol_recovery_full/last_hidden_state_vlast_tlast.json"))
    parser.add_argument("--clean-full-summary", type=Path, default=Path("outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/summary.json"))
    parser.add_argument("--dense-recovery-summary", type=Path, default=Path("outputs/diagnostics/ade20k_dense_protocol_recovery_full/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/ade20k_frequent_class_error_analysis.md"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    write_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
