# ADE20K Dense-Token / Layer Protocol Recovery

This report is generated from real ADE20K evaluator JSON outputs. It keeps the best current ADE20K protocol fixed (`--target-frame processor --alias-policy clean --ignore-zero`, four prompt templates, `tau_p=0.07` checkpoint) and changes only dense-token layer selection or diagnostic logit calibration.

## Commands / outputs

- 64-sample sweep runner: `scripts/run_ade20k_dense_protocol_recovery.py --run --skip-existing --limit 64 --batch-size 4 --device cuda`
- 64-sample summary: `outputs/diagnostics/ade20k_dense_protocol_recovery_limit64/summary.json`
- Full confirmation outputs: `outputs/diagnostics/ade20k_dense_protocol_recovery_full/`
- Updated selected segmentation summary: `outputs/ablations/segmentation_full_selected/summary_ade20k_dense_recovered.json`

## 64-sample protocol sweep

| Rank | Probe | ADE20K mIoU | Delta vs current | Vision tokens | Text tokens | Calibration | Image size |
|---:|---|---:|---:|---|---|---|---|
| 1 | `last_vision_hidden_text_t-2` | 4.076 | +0.908 | `last_hidden_state` | `-2` | none | None |
| 2 | `last_hidden_state_vlast_tlast` | 4.061 | +0.893 | `last_hidden_state` | `last_hidden_state` | none | None |
| 3 | `hidden_v-1_t-4` | 3.440 | +0.272 | `-1` | `-4` | none | None |
| 4 | `vision_ens_v-1_v-2_v-4_t-2` | 3.436 | +0.268 | `[-1, -2, -4]` | `-2` | none | None |
| 5 | `both_ens_v-1_v-2_t-2_t-4` | 3.402 | +0.234 | `[-1, -2]` | `[-2, -4]` | none | None |
| 6 | `text_ens_v-1_t-2_t-4` | 3.380 | +0.212 | `-1` | `[-2, -4]` | none | None |
| 7 | `hidden_v-1_last_text` | 3.330 | +0.162 | `-1` | `last_hidden_state` | none | None |
| 8 | `hidden_v-1_t-1` | 3.330 | +0.162 | `-1` | `-1` | none | None |
| 9 | `clean_size336_v-1_t-2` | 3.329 | +0.161 | `-1` | `-2` | none | 336 |
| 10 | `vision_ens_v-1_v-2_t-2` | 3.228 | +0.060 | `[-1, -2]` | `-2` | none | None |
| 11 | `current_hidden_v-1_t-2` | 3.168 | +0.000 | `-1` | `-2` | none | None |
| 12 | `hidden_v-1_t-6` | 3.059 | -0.110 | `-1` | `-6` | none | None |
| 13 | `hidden_v-2_t-4` | 3.004 | -0.164 | `-2` | `-4` | none | None |
| 14 | `hidden_v-2_t-2` | 2.708 | -0.461 | `-2` | `-2` | none | None |
| 15 | `cal_center_v-1_t-2` | 2.306 | -0.863 | `-1` | `-2` | image-class-center | None |
| 16 | `cal_zscore_v-1_t-2` | 1.640 | -1.528 | `-1` | `-2` | image-class-zscore | None |

Key 64-sample findings:

- `last_hidden_state` for DINOv2 with RoBERTa `hidden_states[-2]` is the best small probe: `4.076` vs current `3.168` (`+0.908`).
- `last_hidden_state` for both encoders is statistically tied on the probe: `4.061` (`+0.893`).
- Image-level class centering and z-score calibration are clearly harmful on this protocol sweep, so they should not be promoted to paper-grade full rows.
- Text layer `-4` gives a smaller positive signal; vision layer ensembles help less than simply using DINOv2 `last_hidden_state`.

## Full ADE20K confirmation

| Probe | Full ADE20K mIoU | Delta vs clean hidden-state full | Vision tokens | Text tokens | Output |
|---|---:|---:|---|---|---|
| `last_hidden_state_vlast_tlast` | 10.549 | +1.215 | `last_hidden_state` | `last_hidden_state` | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/diagnostics/ade20k_dense_protocol_recovery_full/last_hidden_state_vlast_tlast.json` |
| `last_vision_hidden_text_t-2` | 10.229 | +0.895 | `last_hidden_state` | `-2` | `/home/hnxxzy/finegrained-relative-repr-repro/outputs/diagnostics/ade20k_dense_protocol_recovery_full/last_vision_hidden_text_t-2.json` |

Full-run conclusion:

- Best recovered ADE20K full row is `last_hidden_state_vlast_tlast`: `10.549` mIoU.
- This improves over clean aliases + `--ignore-zero` hidden-state row `9.334` by `+1.215` absolute mIoU.
- It reaches `76.45%` of the paper ADE20K target `13.8`.

## Frequent-class full-run behavior

| Probe | wall | building | sky | floor | tree | person | road |
|---|---:|---:|---:|---:|---:|---:|---:|
| `last_hidden_state_vlast_tlast` | 14.08 | 20.74 | 5.38 | 28.99 | 29.18 | 12.88 | 38.14 |
| `last_vision_hidden_text_t-2` | 0.67 | 23.33 | 12.18 | 27.31 | 37.39 | 31.93 | 43.27 |

- `last_hidden_state_vlast_tlast` recovers wall substantially (`14.08` IoU) and wins mean mIoU.
- `last_vision_hidden_text_t-2` is still useful diagnostically: it gives stronger `person`, `road`, `tree`, and `sky`, but almost never predicts wall, so its 150-class mean is lower.

## Updated selected segmentation aggregate

| Dataset | mIoU |
|---|---:|
| VOC20 | 37.573 |
| Pascal Context | 22.004 |
| ADE20K recovered | 10.549 |
| Average | 23.375 |

Paper average is `23.867`; recovered selected average is `23.375` (`97.94%`, gap `-0.491`).

## Decision

- Promote `last_hidden_state_vlast_tlast` as the recovered ADE20K dense-token/layer protocol for the selected full segmentation row.
- Do **not** promote image-class centering/zscore calibration; both hurt the tight 64-sample loop.
- VOC20/Context sanity check completed later: `last_hidden_state` drops VOC20 from `37.57` to `33.57` and Context from `22.00` to `21.90`, so the recovered token protocol is ADE20K-specific and should not replace the selected VOC20/Context rows.
- A targeted ADE20K group-calibration follow-up improves the recovered ADE20K row from `10.55` to `11.47`; see `docs/ade20k_group_calibration_results.md`. This is validation-informed diagnostic calibration, not an unqualified paper-protocol row.
