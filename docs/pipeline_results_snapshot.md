# Ordered Pipeline Results Snapshot

Updated: 2026-07-04 15:20:43 EDT

This file summarizes the completed ordered reproduction pipeline launched via:

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python scripts/run_reproduction_pipeline.py \
  --run \
  --skip-existing \
  --start-at voc20_full_segmentation
```

All pipeline steps reported `completed` except `baseline_tracking_report`, which was skipped because its output already existed.

## Completed stages

- VOC20 full-val segmentation
- Pascal Context segmentation
- ADE20K segmentation
- K sweep: `32, 64, 128, 256, 512`
- `tau_p` sweep: `0.02, 0.05, 0.07, 0.10`; main run covers `0.03`
- token usage ablation: `global`, `mean`, `cap`
- COCO Karpathy anchor-overlap analysis
- baseline tracking report already existed

## Official retrieval comparison

Preferred paper-protocol rows are the Karpathy splits.

| Dataset | Direction | Local R@1 | Paper R@1 | Gap | Relative |
|---|---|---:|---:|---:|---:|
| COCO Karpathy test | I2T | 49.22 | 56.30 | -7.08 | 87.42% |
| COCO Karpathy test | T2I | 36.63 | 42.60 | -5.97 | 85.99% |
| Flickr30k Karpathy test | I2T | 67.40 | 76.30 | -8.90 | 88.34% |
| Flickr30k Karpathy test | T2I | 50.96 | 61.80 | -10.84 | 82.46% |

Official retrieval average R@1: `51.05` vs paper `59.25`, gap `-8.20`, relative `86.17%`.

## Prompt-template sweep

Output: `outputs/prompt_sweep/classification/summary.json`

| Dataset | Best template | Best top-1 | Paper top-1 | Gap | Relative |
|---|---|---:|---:|---:|---:|
| STL10 | `a close-up photo of {class_name}` | 92.15 | 95.30 | -3.15 | 96.69% |
| CIFAR100 | `a photo of {class_name}` | 42.58 | 48.80 | -6.22 | 87.25% |
| Caltech101 | `a close-up photo of {class_name}` | 48.84 | 60.90 | -12.06 | 80.20% |
| DTD | `a cropped photo of {class_name}` | 16.54 | 17.70 | -1.16 | 93.46% |
| EuroSAT | `a close-up photo of {class_name}` | 30.32 | 34.60 | -4.28 | 87.64% |
| Average | mixed best templates | 46.09 | 51.46 | -5.37 | 89.56% |

Prompt sweep improves average top-1 from `44.69` to `46.09` (`+1.40`) but does not close the full paper gap.

## Segmentation comparison

| Dataset | Split | Samples | Local mIoU | Paper mIoU | Gap | Relative |
|---|---|---:|---:|---:|---:|---:|
| VOC20 | val | 1,449 | 14.82 | 32.30 | -17.48 | 45.89% |
| Pascal Context | trainval | 10,103 | 0.53 | 25.50 | -24.97 | 2.09% |
| ADE20K | validation | 2,000 | 1.47 | 13.80 | -12.33 | 10.67% |
| Average | - | - | 5.61 | 23.87 | -18.26 | 23.50% |

Segmentation is now fully executable for all three paper datasets, but the original full-run metrics are historical because the protocol has since been corrected. `docs/segmentation_debug_notes.md` records the follow-up: `scripts/evaluate_segmentation.py` now supports explicit `--target-frame {original,processor}` and `--context-protocol {all459,common59}`. Corrected 16-sample formal probes give VOC20 processor-frame `12.52`, Context common-59 processor-frame `12.31`, and ADE20K processor-frame `0.257`; the Context jump confirms the old 459-class Context path was a major mismatch.

Corrected full-run segmentation results are now available:

| Dataset | Corrected protocol | Samples | Classes | Local mIoU | Paper mIoU | Gap | Relative | Delta vs old |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VOC20 | processor frame | 1,449 | 20 | 20.58 | 32.30 | -11.72 | 63.71% | +5.75 |
| Pascal Context | processor frame + common59 | 10,103 | 59 | 11.23 | 25.50 | -14.27 | 44.05% | +10.70 |
| ADE20K | processor frame | 2,000 | 150 | 2.19 | 13.80 | -11.61 | 15.87% | +0.72 |
| Average | - | - | - | 11.33 | 23.87 | -12.53 | 47.48% | +5.72 |

Outputs:

- `outputs/pal_k512_coco2014_full/voc20_segmentation_full.json`
- `outputs/pal_k512_coco2014_full/context_segmentation_full.json`
- `outputs/pal_k512_coco2014_full/ade20k_segmentation_full.json`
- `outputs/pal_k512_coco2014_full/voc20_segmentation_processor_full.json`
- `outputs/pal_k512_coco2014_full/context_segmentation_common59_processor_full.json`
- `outputs/pal_k512_coco2014_full/ade20k_segmentation_processor_full.json`

## K sweep

The current K sweep records training convergence only; downstream paper metrics per K still need to be run before claiming ablation-table parity.

| K | Final train loss | Train size | Eval size |
|---:|---:|---:|---:|
| 32 | 0.509355 | 82,783 | 0 |
| 64 | 0.403373 | 82,783 | 0 |
| 128 | 0.338583 | 82,783 | 0 |
| 256 | 0.303722 | 82,783 | 0 |
| 512 | 0.283415 | 82,783 | 0 |

Loss decreases monotonically as K increases.

## `tau_p` sweep

The current `tau_p` sweep records training convergence only; downstream metrics per temperature still need to be run for paper-table parity.

| `tau_p` | Final train loss |
|---:|---:|
| 0.02 | 0.252805 |
| 0.03 | 0.283415 |
| 0.05 | 0.335997 |
| 0.07 | 0.371047 |
| 0.10 | 0.407706 |

Lower `tau_p` gives lower training loss in this run, but retrieval/classification/segmentation metrics are required before selecting a paper-grade temperature.

## Token usage ablation

The current token-usage ablation records training convergence only; downstream metrics per pooling mode still need to be run for paper-table parity.

| Mode | Final train loss | Notes |
|---|---:|---|
| global | 0.740161 | first-token/global representation |
| mean | 0.501132 | full-token mean over token-anchor similarities |
| cap | 0.283415 | Cross-Attention Pooling; main PAL setting |

CAP remains best by training loss, consistent with the paper's qualitative claim, but downstream ablation metrics remain necessary.

## Anchor overlap

Output: `outputs/analysis/coco_karpathy_anchor_overlap.json`

| Metric | Value |
|---|---:|
| matched hard overlap | 0.517633 |
| matched Dice | 0.517633 |
| mismatched hard overlap | 0.436705 |
| mismatched Dice | 0.436705 |

Matched vs mismatched overlap gap: `+0.080928` absolute, `+18.53%` relative over mismatched.

## Remaining paper-parity work

1. Run downstream retrieval/classification/segmentation metrics for each K, `tau_p`, and token-usage checkpoint; current sweep outputs are training-loss-only.
2. Expand the CKA proxy result into layer-specific token extraction + retraining if strict layer-selection parity is required.
3. Continue dense segmentation debugging from the corrected full rerun; old segmentation JSONs are baseline/historical.
   - Start from `docs/segmentation_debug_notes.md` and `scripts/diagnose_segmentation_protocol.py`.
   - Corrected full rerun is complete: VOC20 `20.58`, Context `11.23`, ADE20K `2.19`, average `11.33`.
   - ADE20K still needs prompt/name/layer debugging; Context may need dense-token layer selection and prompt ensemble to approach the paper row.
4. Add qualitative attention visualizations.
5. Implement or port baseline rows (CSA, LinearRS, MLPRS, SAIL, FA) if reproducing full comparison tables.
