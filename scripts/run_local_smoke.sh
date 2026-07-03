#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/home/hnxxzy/miniconda3/envs/ovvs/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pal_repro.train \
  --config "$ROOT/configs/pal_strict.yaml" \
  --preset smoke \
  --output-dir "$ROOT/outputs/local_smoke" \
  "$@"
