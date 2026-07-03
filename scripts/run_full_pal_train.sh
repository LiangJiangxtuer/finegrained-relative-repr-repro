#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/home/hnxxzy/miniconda3/envs/ovvs/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pal_repro.train \
  --config "$ROOT/configs/pal_strict.yaml" \
  --data-dir "$ROOT/data/tokens/coco2014_full" \
  --output-dir "$ROOT/outputs/pal_k512_coco2014_full" \
  --num-anchors 512 \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --train-size "${TRAIN_SIZE:-82783}" \
  "$@"
