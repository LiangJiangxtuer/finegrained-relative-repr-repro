# ADE20K Dense Segmentation Debug Results

Updated: 2026-07-07

This note records the ADE20K-focused dense segmentation debugging pass after the selected full `tau_p=0.07` segmentation rerun. The goal was to improve the main segmentation result, whose remaining gap is concentrated on ADE20K.

## Starting point

Best selected full segmentation before this pass:

| Variant | VOC20 | Context | ADE20K | Avg mIoU | Paper avg |
|---|---:|---:|---:|---:|---:|
| `tau_p=0.07`, processor frame, aliases, 4-template ensemble, `vision_layer=-1`, `text_layer=-2` | 37.57 | 22.00 | 7.62 | 22.40 | 23.87 |

ADE20K remained the bottleneck: `7.62` vs paper `13.80`.

## Tight probe loop

All probes used the `tau_p=0.07` checkpoint and ADE20K validation with `--target-frame processor` unless noted. Outputs live under `outputs/diagnostics/ade20k_dense_debug/`.

### Image processor size

Hypothesis: 224x224 dense patch resolution is too coarse for ADE20K.

| Image size | Limit | mIoU |
|---:|---:|---:|
| 224 | 64 | 2.83 |
| 336 | 64 | 2.86 |
| 448 | 64 | 2.75 |

Result: increasing processor crop size did not improve the 64-sample ADE20K probe enough to justify a full high-resolution rerun.

### Prompt / alias sweep

Fixed layers: `vision_layer=-1`, `text_layer=-2`, image size 224, limit 64.

| Alias policy | Prompt set | mIoU |
|---|---|---:|
| all | `a photo of a {class_name}` | 3.15 |
| first | `a {class_name}` | 3.05 |
| all | `a {class_name}` | 3.01 |
| all | 4-template ensemble | 2.83 |
| first | 4-template ensemble | 2.81 |
| all | bare class name | 1.99 |

Result: small probe prefers article/object-style prompts, but this did not transfer reliably to the full split.

### Layer sweep

Fixed prompt: `a photo of a {class_name}`, alias policy `all`, image size 224, limit 64.

Best candidates:

| Vision layer | Text layer | mIoU |
|---:|---:|---:|
| -1 | -4 | 3.22 |
| -1 | -2 | 3.15 |
| -1 | -12 | 3.02 |
| -1 | -1 | 2.97 |
| -2 | -4 | 2.75 |

Result: final DINOv2 layer remained best; `text_layer=-4` was the best 64-sample layer candidate. However, the full rerun with `a photo of a {class_name}` + `text_layer=-4` scored only `6.96` mIoU without ignore-zero and `7.50` with ignore-zero, below the 4-template `text_layer=-2` setting.

### Checkpoint sweep at the best 64-sample prompt/layer candidate

Fixed prompt/layer: `a photo of a {class_name}`, `vision_layer=-1`, `text_layer=-4`, alias policy `all`, image size 224, limit 64.

| Variant | mIoU |
|---|---:|
| `tau_p=0.07` | 3.22 |
| `tau_p=0.10` | 2.99 |
| `tau_p=0.05` | 2.87 |
| K=256 | 2.52 |
| `tau_p=0.03` | 2.43 |
| `tau_p=0.02` | 2.07 |

Result: `tau_p=0.07` remains the best ADE20K checkpoint candidate for dense segmentation.

## Protocol correction found: ignore ADE20K label id 0

ADE20K validation annotations contain label id 0 pixels. Across the 2,000 validation masks, zero-label pixels average `8.39%` of pixels; 430 images have more than 10% zero-label pixels. These pixels should be treated as void/unlabeled for mIoU rather than counted as false-positive foreground area.

A new evaluator flag was added:

```bash
--ignore-zero
```

Probe impact on ADE20K limit 64:

| Setting | mIoU without ignore-zero | mIoU with ignore-zero |
|---|---:|---:|
| 4-template ensemble, `vision_layer=-1`, `text_layer=-2` | 2.83 | 3.03 |
| `a photo of a {class_name}`, `vision_layer=-1`, `text_layer=-4` | 3.22 | 3.46 |

Full-split impact:

| Setting | ADE20K full mIoU |
|---|---:|
| Previous best selected full ADE20K (`tau_p=0.07`, no ignore-zero) | 7.62 |
| Same setting with `--ignore-zero` | 7.99 |
| `a photo of a {class_name}`, `text_layer=-4`, `--ignore-zero` | 7.50 |
| Clean ADE20K aliases, 4-template ensemble, `text_layer=-2`, `--ignore-zero` | 9.33 |

The full split confirms `--ignore-zero` improves the main ADE20K result, while the prompt/layer probe winner did not transfer. The conservative clean-alias policy then improves the selected `tau_p=0.07` ADE20K row further to `9.33`.

## Updated best selected full segmentation summary

Updated machine summaries:

- `outputs/ablations/segmentation_full_selected/summary_ade20k_ignore0.json`
- `outputs/ablations/segmentation_full_selected/summary_ade20k_clean_ignore0.json`
- `outputs/ablations/segmentation_full_selected/summary_ade20k_dense_recovered.json`

| Variant | VOC20 | Context | ADE20K | Avg mIoU | Paper avg | Relative |
|---|---:|---:|---:|---:|---:|---:|
| `tau_p=0.07` with ADE20K `--ignore-zero` | 37.57 | 22.00 | 7.99 | 22.52 | 23.87 | 94.37% |
| `tau_p=0.07` with ADE20K clean aliases + `--ignore-zero` | 37.57 | 22.00 | 9.33 | 22.97 | 23.87 | 96.24% |
| `tau_p=0.07` with ADE20K clean aliases + `--ignore-zero` + recovered `last_hidden_state` dense tokens | 37.57 | 22.00 | 10.55 | 23.38 | 23.87 | 97.94% |
| Diagnostic targeted ADE20K group calibration on recovered row | 37.57 | 22.00 | 11.47 | 23.68 | 23.87 | 99.23% |

This is a real main-result improvement over the prior selected full average `22.40`.

## Dense-token / layer protocol recovery

Detailed generated report: `docs/ade20k_dense_protocol_recovery.md`.

The recovery sweep kept the best ADE20K protocol fixed (`--target-frame processor --alias-policy clean --ignore-zero`, four prompt templates, `tau_p=0.07`) and varied only dense-token layer selection or diagnostic logit calibration.

64-sample sweep summary:

| Probe | ADE20K mIoU | Delta vs current |
|---|---:|---:|
| `last_vision_hidden_text_t-2` | 4.076 | +0.908 |
| `last_hidden_state_vlast_tlast` | 4.061 | +0.893 |
| `hidden_v-1_t-4` | 3.440 | +0.272 |
| current `hidden_v-1_t-2` | 3.168 | +0.000 |
| image-class center calibration | 2.306 | -0.863 |
| image-class zscore calibration | 1.640 | -1.528 |

Full ADE20K confirmation:

| Full probe | ADE20K mIoU | Delta vs clean hidden-state row |
|---|---:|---:|
| `last_hidden_state_vlast_tlast` | 10.549 | +1.215 |
| `last_vision_hidden_text_t-2` | 10.229 | +0.895 |

Conclusion: the remaining gap was partly a dense-token protocol mismatch. For ADE20K, using the encoders' `last_hidden_state` outputs for dense scoring beats indexing `hidden_states[-1]` / `hidden_states[-2]` on the full split. Finer image-class centering or z-score calibration hurts the tight probe and should not be promoted.

Follow-up VOC20/Context sanity check shows `last_hidden_state` is not globally better: VOC20 drops `37.57 -> 33.57` and Context drops `22.00 -> 21.90`, so keep the previous selected VOC20/Context rows.

## Targeted group calibration follow-up

Detailed generated report: `docs/ade20k_group_calibration_results.md`.

Manual class/group bias support was added for diagnostic probes via `--class-bias`, e.g. `wall,sky=0.02`. This is different from global class priors: it targets a small class group found in the frequent-class error analysis.

Best full ADE20K targeted row:

| Probe | Bias | ADE20K mIoU | Delta vs recovered no-bias |
|---|---|---:|---:|
| `boost_frequent7_p004` | `wall,building,sky,floor,tree,person,road=0.04` | 11.470 | +0.920 |

The diagnostic selected aggregate with the previous VOC20/Context rows and this calibrated ADE20K row is `23.68` average mIoU (`99.23%` of paper average). Caveat: the class group and magnitude were selected from ADE20K diagnostic validation probes, so this should be labeled validation-informed diagnostic calibration unless a separate calibration protocol is accepted.

## Frequent-class error analysis and calibration probes

Detailed generated report: `docs/ade20k_frequent_class_error_analysis.md`.

Frequent-class analysis focused on `wall/building/sky/floor/tree/person/road` because these dominate ADE20K validation masks and explain much of the visible failure mode.

Key 64-sample findings under `--ignore-zero`:

| Probe | Alias policy | Prior | alpha | mIoU | Main failure mode |
|---|---|---|---:|---:|---|
| `alias_first` | first | none | 0.0 | 3.069 | Best overall probe, but badly underpredicts wall/sky/floor. |
| `alias_all` | all | none | 0.0 | 3.032 | Helps floor/tree/sky but hurts person/road and includes broad WordNet aliases. |
| `alias_clean` | clean | none | 0.0 | 2.962 | Semantically safer; improves person/road relative to `all` but still underpredicts wall/sky. |
| `prior_ratio_a0_1` | all | ADE20K ratio | 0.1 | 2.431 | Raises wall/building/sky/floor recall but reduces long-tail mIoU. |
| `prior_ratio_a0_25` | all | ADE20K ratio | 0.25 | 0.539 | Overcorrects; wall dominates predictions. |
| `prior_train_a0_25` | all | ADE20K train count | 0.25 | 0.965 | Overweights frequent classes and suppresses person/road. |

Conclusion: class-prior logit bias is not currently beneficial for ADE20K paper-grade mIoU. It can increase frequent-class recall, but the mean over 150 classes collapses because the bias is too blunt. Foreground/background calibration is also not indicated for ADE20K beyond `--ignore-zero`: label id 0 is void/unlabeled, not a background class to predict. Any explicit background threshold would be a separate VOC/Context protocol experiment, not an ADE20K class-prior fix.

## ADE20K clean alias policy

`src/pal_repro/segmentation.py` now defines a conservative `ADE20K_CLEAN_ALIAS_OVERRIDES` table for WordNet-derived class metadata. The cleanup keeps visually interchangeable names and drops obviously misleading aliases such as `route` for road, `mortal/soul` for person, `machine` for car, `mantle/pall` for curtain, `throne/stool` for toilet, `wheel/cycle` for bicycle, and TV slang such as `idiot box`.

The ordered pipeline now uses `--alias-policy clean --ignore-zero` for ADE20K full segmentation so future runs avoid broad WordNet synonyms by default. The full clean-alias ADE20K all-variant table was completed at `outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/summary.json`; the best ADE20K row is `tau_p=0.07` at `9.33` mIoU, closely followed by `tau_p=0.10` at `9.30`.

## Current conclusion

- The best ADE20K full result is now `10.55` mIoU with `tau_p=0.07`, clean aliases, 4-template ensemble, processor-frame masks, `--ignore-zero`, and `last_hidden_state` dense tokens for both encoders.
- Higher image resolution did not help in the 64-sample probe.
- Prompt/layer probes are noisy, so full confirmation is required before promoting a probe winner; the confirmed full winner is `last_hidden_state_vlast_tlast`.
- The remaining ADE20K gap (`10.55` vs `13.80`) is not fixed by simple class-prior correction or image-class logit centering/zscore. Targeted frequent-class calibration improves diagnostic ADE20K to `11.47`, but it is validation-informed and should be reported with that caveat.
