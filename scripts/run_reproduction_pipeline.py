#!/usr/bin/env python3
"""List or run the remaining PAL paper-reproduction pipeline steps."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from pal_repro.pipeline import PipelineConfig, PipelineStep, build_pipeline_steps


def select_steps(
    steps: list[PipelineStep],
    only: list[str],
    start_at: str | None,
    until: str | None,
) -> list[PipelineStep]:
    selected = steps
    if start_at is not None:
        names = [step.name for step in selected]
        if start_at not in names:
            raise KeyError(f"unknown start step {start_at!r}")
        selected = selected[names.index(start_at) :]
    if until is not None:
        names = [step.name for step in selected]
        if until not in names:
            raise KeyError(f"unknown until step {until!r}")
        selected = selected[: names.index(until) + 1]
    if only:
        wanted = set(only)
        selected = [step for step in selected if step.name in wanted]
        missing = wanted - {step.name for step in selected}
        if missing:
            raise KeyError(f"unknown selected steps: {sorted(missing)}")
    return selected


def step_to_json(step: PipelineStep) -> dict[str, object]:
    return {
        "name": step.name,
        "priority": step.priority,
        "command": step.command,
        "output": None if step.output is None else str(step.output),
        "description": step.description,
        "gpu": step.gpu,
        "long_running": step.long_running,
    }


def run_steps(steps: list[PipelineStep], root: Path, skip_existing: bool, dry_run: bool) -> list[dict[str, object]]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    results: list[dict[str, object]] = []
    for step in steps:
        if skip_existing and step.output is not None and step.output.exists():
            print(f"SKIP existing {step.name}: {step.output}", flush=True)
            results.append({"name": step.name, "status": "skipped", "output": str(step.output)})
            continue
        print(f"RUN {step.name}: {' '.join(step.command)}", flush=True)
        if dry_run:
            results.append({"name": step.name, "status": "dry_run"})
            continue
        completed = subprocess.run(step.command, cwd=root, env=env, check=False)
        status = "completed" if completed.returncode == 0 else "failed"
        row = {"name": step.name, "status": status, "returncode": completed.returncode}
        results.append(row)
        if completed.returncode != 0:
            raise SystemExit(f"pipeline step failed: {step.name} returncode={completed.returncode}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = PipelineConfig(root=args.root)
    steps = select_steps(build_pipeline_steps(config), args.only, args.start_at, args.until)
    if args.list or not args.run:
        print(json.dumps([step_to_json(step) for step in steps], indent=2, sort_keys=True))
    if args.run:
        results = run_steps(steps, root=args.root, skip_existing=args.skip_existing, dry_run=args.dry_run)
        print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
