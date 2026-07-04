#!/usr/bin/env python3
"""Write the baseline-reproduction tracking report for complete paper tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINES = {
    "CSA": {
        "status": "not_implemented",
        "reason": "Requires exact paper baseline implementation/protocol; not present in current strict PAL package.",
    },
    "LinearRS": {
        "status": "available_in_reference_scaffold_only",
        "reason": "Reference bridge-anchors has linear-style modules, but paper-exact training/eval wiring is not yet ported.",
    },
    "MLPRS": {
        "status": "available_in_reference_scaffold_only",
        "reason": "Reference bridge-anchors has MLP-style modules, but paper-exact training/eval wiring is not yet ported.",
    },
    "SAIL": {
        "status": "not_implemented",
        "reason": "Needs official or reimplemented SAIL baseline before comparison-row parity can be claimed.",
    },
    "FA": {
        "status": "not_implemented",
        "reason": "Needs fixed-anchor baseline wiring and downstream evaluations.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "purpose": "Track non-PAL baseline work required for complete paper-table reproduction.",
        "baselines": BASELINES,
        "next_steps": [
            "Identify exact baseline hyperparameters and data protocol from paper/appendix or upstream code.",
            "Port each baseline behind the same train/eval token-cache interface as PAL.",
            "Run the same retrieval/classification/segmentation and ablation matrix before claiming full table parity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
