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
- Added an ordered, executable paper-reproduction pipeline:
  - `src/pal_repro/pipeline.py`
  - `scripts/run_reproduction_pipeline.py`
  - `scripts/prepare_karpathy_splits.py`
  - `scripts/extract_karpathy_retrieval_tokens.py`
  - `scripts/run_cka_layer_sweep.py`
  - `scripts/run_prompt_sweep.py`
  - `scripts/analyze_anchor_overlap.py`
  - `scripts/write_baseline_tracking_report.py`
- Added segmentation loaders for Pascal Context and ADE20K metadata-backed evaluation, plus explicit dense-eval protocol flags for processor-frame scoring and Pascal Context common-59 vs all-459 evaluation.
- Added token-usage ablation support via `pooling_mode={cap,mean,global}`.

## Verification evidence

Canonical local test command:

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
```

Result:

```text
Ran 63 tests in 0.206s
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

Karpathy official split alignment is now complete for COCO and Flickr30k:

```text
split metadata: data/splits/karpathy/dataset_coco.json and dataset_flickr30k.json
COCO Karpathy tokens: data/tokens/coco2014_karpathy_test_multicaption
COCO Karpathy images/texts: 5000 / 25010
COCO Karpathy output: outputs/pal_k512_coco2014_full/coco_karpathy_test_multicaption_retrieval.json
COCO Karpathy I2T R@1/R@5/R@10: 49.22 / 77.70 / 86.56
COCO Karpathy T2I R@1/R@5/R@10: 36.63 / 66.37 / 77.87

Flickr30k Karpathy tokens: data/tokens/flickr30k_karpathy_test_multicaption
Flickr30k Karpathy images/texts: 1000 / 5000
Flickr30k Karpathy output: outputs/pal_k512_coco2014_full/flickr30k_karpathy_test_multicaption_retrieval.json
Flickr30k Karpathy I2T R@1/R@5/R@10: 67.40 / 90.10 / 94.90
Flickr30k Karpathy T2I R@1/R@5/R@10: 50.96 / 79.18 / 86.76
```

A CKA proxy sweep over layers `[-1, -2, -6, -12]` on 128 COCO Karpathy pairs has also completed:

```text
output: outputs/cka/coco_karpathy_layer_sweep.json
best pair: vision_layer=-1, text_layer=-2, linear CKA=0.665336
```

Prompt-template classification sweep has completed:

```text
output: outputs/prompt_sweep/classification/summary.json
best average top1: 46.09 vs paper 51.46
gap: -5.37, relative: 89.56%
best templates: STL10 close-up, CIFAR100 default photo, Caltech101 close-up, DTD cropped, EuroSAT close-up
```

VOC20/Context/ADE20K full segmentation has completed:

```text
VOC20 output: outputs/pal_k512_coco2014_full/voc20_segmentation_full.json
VOC20 samples/mIoU: 1449 / 14.82 vs paper 32.30
Context output: outputs/pal_k512_coco2014_full/context_segmentation_full.json
Context samples/mIoU: 10103 / 0.53 vs paper 25.50
ADE20K output: outputs/pal_k512_coco2014_full/ade20k_segmentation_full.json
ADE20K samples/mIoU: 2000 / 1.47 vs paper 13.80
Average mIoU: 5.61 vs paper 23.87, relative 23.50%
```

The downstream pipeline process `proc_fd7c67b922d5` completed normally. It also completed training-only K, `tau_p`, and token-usage sweeps plus anchor-overlap analysis:

```text
K final train loss: K32 0.509355 / K64 0.403373 / K128 0.338583 / K256 0.303722 / K512 0.283415
tau_p final train loss: 0.02 0.252805 / 0.03 0.283415 / 0.05 0.335997 / 0.07 0.371047 / 0.10 0.407706
token usage final train loss: global 0.740161 / mean 0.501132 / cap 0.283415
anchor overlap: matched 0.517633 vs mismatched 0.436705
```

Full pipeline summary: `docs/pipeline_results_snapshot.md`.

Segmentation protocol debug and formal evaluator update are complete:

```text
script: scripts/diagnose_segmentation_protocol.py
notes: docs/segmentation_debug_notes.md
probe outputs: outputs/diagnostics/*_segmentation_protocol_probe*.json
formal evaluator: scripts/evaluate_segmentation.py --target-frame {original,processor} --context-protocol {all459,common59}
corrected probes: VOC20 processor 12.5179; Context common59 processor 12.3125; ADE20K processor 0.2571
finding: DINOv2 image preprocessing uses resize-shortest-edge 256 plus 224x224 center crop; the old full segmentation metrics scored against original masks and Context used all 459 labels. Context common59 + processor-frame scoring fixes a major protocol mismatch; ADE20K remains an active prompt/name/layer-selection suspect.
```

Corrected full segmentation rerun completed normally:

```text
process: proc_a795405ceceb, exit_code=0
log: outputs/logs/corrected_segmentation_rerun_20260705_104725.log
VOC20 corrected:   outputs/pal_k512_coco2014_full/voc20_segmentation_processor_full.json, mIoU 20.58 vs paper 32.30
Context corrected: outputs/pal_k512_coco2014_full/context_segmentation_common59_processor_full.json, mIoU 11.23 vs paper 25.50
ADE20K corrected:  outputs/pal_k512_coco2014_full/ade20k_segmentation_processor_full.json, mIoU 2.19 vs paper 13.80
Corrected average: 11.33 vs paper 23.87; historical average was 5.61
```
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

- Downstream retrieval and full classification metrics are complete for the K, `tau_p`, and token-usage ablation checkpoints (`docs/ablation_downstream_retrieval_results.md`, `docs/ablation_downstream_classification_segmentation_results.md`). Corrected segmentation has 64-sample probes for all sweep checkpoints, selected full reruns for K=256 / `tau_p=0.07`, full ADE20K clean-alias rows for every sweep checkpoint, ADE20K dense-token/layer recovery, and targeted group-calibration diagnostics; the best uncalibrated selected full rerun is `tau_p=0.07` with VOC20 `37.57`, Context `22.00`, ADE20K `10.55`, average `23.38` vs paper `23.87`. Diagnostic targeted ADE20K calibration reaches ADE20K `11.47`, average `23.68` (`99.23%`) but is validation-informed.
- If desired, expand the CKA proxy sweep into full layer-specific token extraction/training/evaluation.
- Dense segmentation debugging now has an ADE20K recovered `last_hidden_state` dense-token full confirmation and targeted group-calibration diagnostics. VOC20/Context sanity check shows `last_hidden_state` should stay ADE20K-specific (`37.57 -> 33.57` for VOC20, `22.00 -> 21.90` for Context). Simple class priors plus image-class center/zscore calibration did not help; targeted frequent-class bias helps ADE20K but needs a held-out calibration protocol if it is to be treated as more than diagnostic.
- Produce anchor-overlap and qualitative attention visualizations.
- Implement/run baseline methods if reproducing every comparison row, not just PAL target rows.
