# Downstream Retrieval Ablation Results

Updated: 2026-07-06

This file records the next-stage downstream retrieval evaluation for already-trained ablation checkpoints. Metrics are computed on the official/Karpathy COCO test and Flickr30k test multi-caption token caches. `Avg R@1` is the mean of four R@1 values: COCO I2T, COCO T2I, Flickr30k I2T, and Flickr30k T2I.

Execution note: the live CUDA probe in this session returned `torch.cuda.is_available() == False` and `nvidia-smi` could not communicate with the driver, so these cached-token retrieval evaluations were run on CPU via `--device cpu`. They do not run frozen DINOv2/RoBERTa extraction and are therefore still practical without GPU.

## Output artifacts

- Per-run JSONs: `outputs/ablations/retrieval/*_karpathy_test_multicaption_retrieval.json`
- Machine-readable summary: `outputs/ablations/retrieval/summary.json`

## Anchor count K retrieval sweep

| K | COCO I2T R@1 | COCO T2I R@1 | Flickr30k I2T R@1 | Flickr30k T2I R@1 | Avg R@1 |
|---:|---:|---:|---:|---:|---:|
| 32 | 33.56 | 24.73 | 49.50 | 35.98 | 35.94 |
| 64 | 39.74 | 29.18 | 55.50 | 41.56 | 41.50 |
| 128 | 43.72 | 33.22 | 61.50 | 46.64 | 46.27 |
| 256 | 47.26 | 35.85 | 65.40 | 50.22 | 49.68 |
| 512 | 49.22 | 36.63 | 67.40 | 50.96 | 51.05 |

Finding: retrieval improves monotonically with K in this local run (`35.94 -> 51.05` Avg R@1 from K=32 to K=512), matching the paper's qualitative anchor-count trend even though the PDF text does not expose exact numeric K points.

## CAP temperature retrieval sweep

| tau_p | COCO I2T R@1 | COCO T2I R@1 | Flickr30k I2T R@1 | Flickr30k T2I R@1 | Avg R@1 | Paper avg_ret | Gap | Relative |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 50.24 | 37.23 | 67.00 | 51.16 | 51.41 | 58.80 | -7.39 | 87.43% |
| 0.03 | 49.22 | 36.63 | 67.40 | 50.96 | 51.05 | 59.30 | -8.25 | 86.09% |
| 0.05 | 46.92 | 35.37 | 64.10 | 49.88 | 49.07 | 57.90 | -8.83 | 84.75% |
| 0.07 | 44.50 | 33.96 | 61.90 | 47.18 | 46.88 | 55.50 | -8.62 | 84.48% |
| 0.10 | 41.80 | 32.00 | 59.30 | 45.22 | 44.58 | 52.90 | -8.32 | 84.27% |

Finding: `tau_p=0.02` is best for retrieval in this local run (`51.41` Avg R@1), slightly above the main `0.03` run (`51.05`). This mirrors the lower training loss at `0.02`, but paper-average classification and segmentation metrics are still required before changing the main setting.

## Token usage retrieval sweep

| Mode | COCO I2T R@1 | COCO T2I R@1 | Flickr30k I2T R@1 | Flickr30k T2I R@1 | Avg R@1 | Paper avg_ret | Gap | Relative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global only | 21.66 | 15.60 | 37.70 | 25.82 | 25.20 | 43.90 | -18.70 | 57.39% |
| full tokens mean | 34.14 | 25.96 | 50.60 | 38.34 | 37.26 | 48.40 | -11.14 | 76.98% |
| full tokens CAP | 49.22 | 36.63 | 67.40 | 50.96 | 51.05 | 59.30 | -8.25 | 86.09% |

Finding: CAP is strongly better than mean/global for retrieval (`51.05` vs `37.26` vs `25.20` Avg R@1), supporting the paper's token-CAP claim on downstream retrieval rather than train loss only. Absolute local averages are still below paper avg_ret targets.

## Remaining ablation work

- Downstream classification is now complete for the same K / `tau_p` / token-usage checkpoints; see `docs/ablation_downstream_classification_segmentation_results.md`.
- Corrected segmentation has 64-sample probes for the same checkpoints, selected full VOC20/Context reruns, full ADE20K clean-alias rows, ADE20K dense-token recovery, and targeted group-calibration diagnostics. Only the remaining full VOC20/Context checkpoint rows are optional for full ablation-table parity.
