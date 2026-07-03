#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/home/hnxxzy/miniconda3/envs/ovvs/bin/python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$ROOT/scripts/extract_coco_tokens.py" \
  --captions-json /home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/coco2014/raw/annotations/captions_train2014.json \
  --image-dir /home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/coco2014/raw/train2014 \
  --output-dir "$ROOT/data/tokens/coco2014_full" \
  --vision-model facebook/dinov2-large \
  --text-model roberta-large \
  --batch-size "${BATCH_SIZE:-8}" \
  --max-text-length "${MAX_TEXT_LENGTH:-64}" \
  --storage-dtype float16 \
  --output-format chunks \
  --chunk-size "${CHUNK_SIZE:-2048}" \
  "$@"
