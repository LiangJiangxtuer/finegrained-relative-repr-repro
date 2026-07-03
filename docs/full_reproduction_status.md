# Full Reproduction Status

## Scope correction

The target is not a smoke demo. The reproduction target is now the complete paper code path: paper claims, formulas, engineering choices, datasets, main experiments, ablations, and analyses, using `shiwonkim/bridge-anchors` as the scaffold/base.

## Completed in this pass

- Wrote paper-level claim/formula/experiment analysis:
  - `docs/paper_claims_and_experiments.md`
- Wrote full implementation plan:
  - `docs/full_reproduction_plan.md`
- Added the paper experiment/target matrix:
  - `configs/reproduction_matrix.yaml`
  - `src/pal_repro/experiment_matrix.py`
- Added local data manifest:
  - `configs/data_manifest.local.yaml`
- Extended PAL core for downstream paper evaluations:
  - `ProjectionFreeAnchorLearning.encode_image`
  - `ProjectionFreeAnchorLearning.encode_text`
- Added paper-task evaluation primitives:
  - retrieval R@K percentages via `evaluate_retrieval_model`
  - zero-shot classification via `evaluate_zero_shot_classification`
  - foreground mIoU via `foreground_miou`
  - anchor overlap/Dice via `anchor_overlap_report`
- Made full-cache data loading less memory-wasteful:
  - `load_token_tensors` preserves fp16 storage dtype;
  - training casts batches to fp32 only on device.
- Upgraded COCO token extraction script for paper-scale use:
  - fixed max-length padding by default;
  - fp16 storage;
  - chunked output mode;
  - metadata and pairs JSONL output.
- Added full-run helper scripts:
  - `scripts/run_full_coco_extraction.sh`
  - `scripts/merge_token_chunks.py`
  - `scripts/run_full_pal_train.sh`

## Verification evidence

Canonical local test command:

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
```

Result:

```text
Ran 15 tests in 1.272s
OK
```

Ad-hoc focused verification script:

```text
/tmp/hermes-verify-mj33pzbf.py
```

Result:

```text
AD_HOC_VERIFICATION_OK paper contract, PAL eval hooks, fp16 data path, tiny train path
exit_code=0
cleaned /tmp/hermes-verify-mj33pzbf.py
```

## Environment changes

Installed into `/home/hnxxzy/miniconda3/envs/ovvs`:

- `transformers 4.57.6`
- `scikit-learn 1.9.0`

Verified environment:

- torch 2.5.1+cu124
- CUDA available
- GPU: NVIDIA GeForce RTX 4090, 24GB
- free disk on `/`: about 244GB at probe time

## Current long task

The previous blocker is resolved: `facebook/dinov2-large` finished downloading to the HuggingFace cache, and a one-sample local-files-only probe succeeded.

Probe output:

```text
outputs/extract_tiny_probe_20260701_202238
image_tokens.pt: (1, 257, 1024) float16
text_tokens.pt: (1, 64, 1024) float16
text_mask.pt: (1, 64) bool
```

Full COCO2014 token extraction is now running in the background:

```text
process id: proc_ba6df0a318de
command: PYTHONUNBUFFERED=1 BATCH_SIZE=8 CHUNK_SIZE=2048 bash scripts/run_full_coco_extraction.sh
```

Early progress evidence:

```text
processed 4072/82783
chunks written under data/tokens/coco2014_full/chunks/
current cache size at probe time: 3.8G
```

## Next execution step after extraction completes

Full COCO2014 token extraction and merge are complete.

Merged token cache:

```text
image_tokens.pt: (82783, 257, 1024) float16, 40.58 GiB
text_tokens.pt:  (82783, 64, 1024)  float16, 10.11 GiB
text_mask.pt:    (82783, 64)        bool
metadata_merged.json format: monolithic_from_chunks
```

A K=512 full PAL training run has been started in the background:

```text
process id: proc_7b5dbc6769c6
command: PYTHONUNBUFFERED=1 EPOCHS=20 BATCH_SIZE=128 TRAIN_SIZE=82783 bash scripts/run_full_pal_train.sh
```

Early resource probe while training:

```text
GPU: ~1351 MiB used, ~42% utilization
CPU RAM: ~79 GiB used, ~43 GiB available
```

The trainer loaded the full fp16 token cache and trained on all 82,783 COCO2014 first-caption pairs. The run completed normally:

```text
process id: proc_7b5dbc6769c6
final_train_loss: 0.28341474021328655
checkpoint: outputs/pal_k512_coco2014_full/checkpoint.pt
metrics: outputs/pal_k512_coco2014_full/metrics.json
trainable params: anchors_img, anchors_txt
```

Because `TRAIN_SIZE=82783`, this run used all COCO training pairs and intentionally has no internal held-out retrieval split (`eval_size: 0`). Retrieval evaluation has now produced two COCO validation results:

```text
first-caption output: outputs/pal_k512_coco2014_full/coco_val_first_caption_retrieval.json
first-caption samples: 40504
first-caption I2T R@1/R@5/R@10: 15.22 / 33.36 / 43.77
first-caption T2I R@1/R@5/R@10: 15.01 / 33.73 / 43.83

5K multi-caption output: outputs/pal_k512_coco2014_full/coco_val_5k_multicaption_retrieval.json
5K multi-caption images/texts: 5000 / 25021
5K multi-caption I2T R@1/R@5/R@10: 49.82 / 77.66 / 86.68
5K multi-caption T2I R@1/R@5/R@10: 37.06 / 66.29 / 77.80
paper COCO R@1 target: I2T 56.3 / T2I 42.6
```

The 5K multi-caption protocol is much closer to the paper setting and reaches ~88.5% of the paper I2T R@1 and ~87.0% of the paper T2I R@1. It still uses the local seed=42 5K subset; exact parity requires identifying the paper's split if it is fixed.

Flickr30k local 1K multi-caption retrieval has also been evaluated from `/home/hnxxzy/Downloads/Flickr30k.zip`:

```text
output: outputs/pal_k512_coco2014_full/flickr30k_1k_multicaption_retrieval.json
images/texts: 1000 / 5000
I2T R@1/R@5/R@10: 67.80 / 90.20 / 95.40
T2I R@1/R@5/R@10: 52.80 / 80.62 / 87.52
paper Flickr30k R@1 target: I2T 76.3 / T2I 61.8
```

The local Flickr30k result reaches ~88.9% of the paper I2T R@1 and ~85.4% of the paper T2I R@1; exact parity still requires identifying the official split used by the paper.

Zero-shot classification has also been evaluated:

```text
STL10:      top1 91.96 vs paper 95.30
CIFAR100:   top1 42.58 vs paper 48.80
Caltech101: top1 45.63 vs paper 60.90
DTD:        top1 15.21 vs paper 17.70
EuroSAT:    top1 28.08 vs paper 34.60
Average:    top1 44.69 vs paper 51.46
```

Full machine-readable results are summarized in `docs/results_snapshot.md`.

## Remaining for paper-grade result parity

- Replace deterministic local COCO/Flickr subsets with the exact official/Karpathy split if the paper used one.
- Reproduce or explicitly sweep/document the CKA layer-selection setting; current runs use final hidden tokens.
- Complete VOC20 full-val segmentation and add Pascal Context / ADE20K loaders and full foreground-mIoU evaluation.
- Run ablations for token usage/CAP, anchor count K, CAP temperature `tau_p`, and COCO80K+COCO2017 data scaling.
- Produce anchor-overlap and qualitative attention visualizations.
- Implement/run baseline methods if reproducing every comparison row, not just PAL target rows.
