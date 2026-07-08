# ADE20K Frequent-Class Error Analysis

This report is generated from real evaluator JSON outputs. The frequent classes are `wall/building/sky/floor/tree/person/road`.

## Sources

- 64-sample alias/prior probe summary: `outputs/diagnostics/ade20k_dense_debug/alias_prior_limit64_ignore0/summary.json`
- Selected full ADE20K result: `outputs/diagnostics/ade20k_dense_protocol_recovery_full/last_hidden_state_vlast_tlast.json`
- Clean-alias full ADE20K all-variant summary: `outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/summary.json`
- Dense-token/layer recovery summary: `outputs/diagnostics/ade20k_dense_protocol_recovery_full/summary.json`

## 64-sample alias / prior probe overview

| Probe | Alias policy | Prior | alpha | mIoU | Top predicted class |
|---|---|---|---:|---:|---|
| `alias_first` | first | none | 0.0 | 3.069 | swivel chair |
| `alias_all` | all | none | 0.0 | 3.032 | swivel chair |
| `alias_clean` | clean | none | 0.0 | 2.962 | swivel chair |
| `prior_ratio_a0_1` | all | ade20k-ratio | 0.1 | 2.431 | wall |
| `prior_train_a0_25` | all | ade20k-train-count | 0.25 | 0.965 | wall |
| `prior_train_a0_5` | all | ade20k-train-count | 0.5 | 0.676 | wall |
| `prior_ratio_a0_25` | all | ade20k-ratio | 0.25 | 0.539 | wall |
| `prior_ratio_a0_5` | all | ade20k-ratio | 0.5 | 0.255 | wall |
| `prior_ratio_a0_75` | all | ade20k-ratio | 0.75 | 0.255 | wall |
| `prior_ratio_a1_0` | all | ade20k-ratio | 1.0 | 0.255 | wall |

## Frequent-class detail on the 64-sample probe

Values are IoU / pred-target ratio / recall / precision. A low pred-target ratio means the class is rarely predicted; a high ratio with low precision means the class is overused.

### `alias_first` (alias=first, prior=none, alpha=0.0, mIoU=3.069)

| Class | IoU | pred/target | recall | precision |
|---|---:|---:|---:|---:|
| wall | 0.06 | 0.001 | 0.1% | 100.0% |
| building | 24.87 | 0.346 | 26.8% | 77.5% |
| sky | 1.21 | 0.012 | 1.2% | 99.3% |
| floor | 1.35 | 0.016 | 1.4% | 85.8% |
| tree | 6.90 | 0.138 | 7.3% | 53.2% |
| person | 14.80 | 0.179 | 15.2% | 85.1% |
| road | 31.64 | 0.548 | 37.2% | 67.9% |

### `alias_all` (alias=all, prior=none, alpha=0.0, mIoU=3.032)

| Class | IoU | pred/target | recall | precision |
|---|---:|---:|---:|---:|
| wall | 0.18 | 0.003 | 0.2% | 64.0% |
| building | 16.63 | 0.351 | 19.3% | 54.9% |
| sky | 3.02 | 0.032 | 3.0% | 95.2% |
| floor | 20.46 | 0.491 | 25.3% | 51.6% |
| tree | 9.04 | 0.220 | 10.1% | 46.0% |
| person | 1.94 | 0.027 | 2.0% | 71.7% |
| road | 18.13 | 0.371 | 21.0% | 56.7% |

### `alias_clean` (alias=clean, prior=none, alpha=0.0, mIoU=2.962)

| Class | IoU | pred/target | recall | precision |
|---|---:|---:|---:|---:|
| wall | 0.10 | 0.001 | 0.1% | 68.5% |
| building | 10.46 | 0.158 | 11.0% | 69.4% |
| sky | 1.42 | 0.014 | 1.4% | 98.0% |
| floor | 12.20 | 0.188 | 12.9% | 68.8% |
| tree | 8.36 | 0.193 | 9.2% | 47.6% |
| person | 15.35 | 0.188 | 15.8% | 84.0% |
| road | 34.56 | 0.605 | 41.2% | 68.1% |

### `prior_ratio_a0_1` (alias=all, prior=ade20k-ratio, alpha=0.1, mIoU=2.431)

| Class | IoU | pred/target | recall | precision |
|---|---:|---:|---:|---:|
| wall | 25.69 | 2.806 | 77.8% | 27.7% |
| building | 23.62 | 2.757 | 71.8% | 26.0% |
| sky | 27.10 | 0.610 | 34.3% | 56.2% |
| floor | 37.50 | 0.829 | 49.9% | 60.2% |
| tree | 14.30 | 0.327 | 16.6% | 50.8% |
| person | 0.00 | 0.000 | 0.0% | 0.0% |
| road | 1.04 | 0.033 | 1.1% | 32.4% |

### `prior_ratio_a0_25` (alias=all, prior=ade20k-ratio, alpha=0.25, mIoU=0.539)

| Class | IoU | pred/target | recall | precision |
|---|---:|---:|---:|---:|
| wall | 19.91 | 4.865 | 97.4% | 20.0% |
| building | 19.37 | 0.498 | 24.3% | 48.8% |
| sky | 1.65 | 0.017 | 1.7% | 94.4% |
| floor | 0.01 | 0.000 | 0.0% | 100.0% |
| tree | 0.00 | 0.000 | 0.0% | 0.0% |
| person | 0.00 | 0.000 | 0.0% | 0.0% |
| road | 0.00 | 0.000 | 0.0% | 0.0% |

### `prior_train_a0_25` (alias=all, prior=ade20k-train-count, alpha=0.25, mIoU=0.965)

| Class | IoU | pred/target | recall | precision |
|---|---:|---:|---:|---:|
| wall | 22.16 | 3.382 | 79.5% | 23.5% |
| building | 0.35 | 0.004 | 0.4% | 91.5% |
| sky | 22.47 | 0.343 | 24.6% | 71.9% |
| floor | 20.75 | 3.624 | 79.5% | 21.9% |
| tree | 6.86 | 0.149 | 7.4% | 49.6% |
| person | 0.00 | 0.000 | 0.0% | 0.0% |
| road | 0.00 | 0.000 | 0.0% | 0.0% |

## Selected full ADE20K frequent-class IoU

Selected full result mIoU: `10.549` with alias policy `clean`, ignore index `0`.

| Class | Full IoU |
|---|---:|
| wall | 14.08 |
| building | 20.74 |
| sky | 5.38 |
| floor | 28.99 |
| tree | 29.18 |
| person | 12.88 |
| road | 38.14 |

## Full ADE20K clean-alias rows

These rows use `--alias-policy clean --ignore-zero` and share frozen image forward across ablation checkpoints.

| Variant | Group | Label | ADE20K mIoU | Output |
|---|---|---|---:|---|
| `tau_0_07` | tau | 0_07 | 9.334 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/tau/0_07/ade20k_full_segmentation.json` |
| `tau_0_10` | tau | 0_10 | 9.301 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/tau/0_10/ade20k_full_segmentation.json` |
| `tau_0_05` | tau | 0_05 | 8.663 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/tau/0_05/ade20k_full_segmentation.json` |
| `k_256` | k | 256 | 7.883 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/k/256/ade20k_full_segmentation.json` |
| `k_128` | k | 128 | 7.636 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/k/128/ade20k_full_segmentation.json` |
| `k_512` | k | 512 | 7.512 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/k/512/ade20k_full_segmentation.json` |
| `tau_0_03` | tau | 0_03 | 7.512 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/tau/0_03/ade20k_full_segmentation.json` |
| `token_usage_cap` | token_usage | cap | 7.512 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/token_usage/cap/ade20k_full_segmentation.json` |
| `tau_0_02` | tau | 0_02 | 6.926 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/tau/0_02/ade20k_full_segmentation.json` |
| `k_64` | k | 64 | 6.872 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/k/64/ade20k_full_segmentation.json` |
| `token_usage_mean` | token_usage | mean | 5.895 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/token_usage/mean/ade20k_full_segmentation.json` |
| `k_32` | k | 32 | 4.920 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/k/32/ade20k_full_segmentation.json` |
| `token_usage_global` | token_usage | global | 0.154 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/token_usage/global/ade20k_full_segmentation.json` |

## Dense-token / layer recovery full confirmation

These rows keep clean ADE20K aliases and `--ignore-zero` fixed, then vary dense-token layer selection on the full ADE20K validation split.

| Probe | ADE20K mIoU | Output |
|---|---:|---|
| `last_hidden_state_vlast_tlast` | 10.549 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/diagnostics/ade20k_dense_protocol_recovery_full/last_hidden_state_vlast_tlast.json` |
| `last_vision_hidden_text_t-2` | 10.229 | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/diagnostics/ade20k_dense_protocol_recovery_full/last_vision_hidden_text_t-2.json` |

## Calibration / prior-correction conclusion

- The uncalibrated alias probes remain best on overall mIoU. `alias_first`, `alias_all`, and `alias_clean` are tightly clustered around 3.0 mIoU on the 64-sample probe.
- Class-prior logit biases improve some frequent-class recall (especially wall/sky/floor) but collapse the long-tail mean mIoU: the best tested prior row is below the uncalibrated alias rows.
- The prior bias is too blunt: `ade20k-ratio` with alpha >= 0.25 makes wall dominate predictions, while `ade20k-train-count` similarly overweights frequent classes and suppresses person/road.
- A foreground/background calibration is not indicated for ADE20K beyond `--ignore-zero`: the evaluation has 150 labeled foreground classes and label id 0 should be void/ignored, not a learned background class. For VOC/Context, explicit background calibration could be studied separately, but it would be a protocol change rather than an ADE20K class-prior fix.
- Concrete probe numbers: uncalibrated `alias_first` mIoU `3.069` vs `ade20k-ratio` alpha=0.1 `2.431` and alpha=0.25 `0.539`.
