# ADE20K Targeted Group Calibration Results

This note records the follow-up after ADE20K dense-token recovery. It covers two questions:

1. Does the recovered `last_hidden_state` dense-token protocol transfer to VOC20 / Pascal Context?
2. Can targeted ADE20K class/group calibration improve over the recovered ADE20K `last_hidden_state` row without using blunt global priors or image-class z-scoring?

All numbers below are from real evaluator JSON outputs.

## VOC20 / Context sanity check for recovered `last_hidden_state`

Command output summary: `outputs/diagnostics/dense_protocol_sanity_voc_context_last_hidden/summary.json`.

The sanity check used `tau_p=0.07`, processor-frame masks, the same four segmentation templates, and no explicit `--vision-layer` / `--text-layer`, which means both encoders use `outputs.last_hidden_state`.

| Dataset | Previous selected hidden-index mIoU | `last_hidden_state` mIoU | Delta | Output |
|---|---:|---:|---:|---|
| VOC20 | 37.573 | 33.567 | -4.006 | `outputs/diagnostics/dense_protocol_sanity_voc_context_last_hidden/voc20_last_hidden_full.json` |
| Pascal Context common59 | 22.004 | 21.902 | -0.102 | `outputs/diagnostics/dense_protocol_sanity_voc_context_last_hidden/context_last_hidden_full.json` |

Conclusion: recovered `last_hidden_state` is **ADE20K-specific**, not a global segmentation protocol replacement. Keep the previous selected explicit hidden-state rows for VOC20 / Context, and use `last_hidden_state` only for ADE20K unless a future all-dataset sweep says otherwise.

## Targeted ADE20K group-calibration probe

Runner: `scripts/run_ade20k_group_calibration_probe.py`.

Probe command:

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python \
  scripts/run_ade20k_group_calibration_probe.py \
  --run --skip-existing --limit 64 --batch-size 4 --device cuda
```

Probe summary: `outputs/diagnostics/ade20k_group_calibration_limit64/summary.json`.

The probe keeps the recovered ADE20K protocol fixed:

- `tau_p=0.07` checkpoint
- DINOv2 / RoBERTa `last_hidden_state` dense tokens
- processor-frame masks
- conservative clean aliases
- `--ignore-zero`
- four-template segmentation prompt ensemble

It only adds explicit class/group logit biases via `--class-bias`, e.g. `wall,sky=0.02`. These are diagnostic manual biases, not global priors.

Top 64-sample rows:

| Probe | Class bias | mIoU | Delta vs no-bias baseline |
|---|---|---:|---:|
| `boost_frequent7_p004` | `wall,building,sky,floor,tree,person,road=0.04` | 4.658 | +0.596 |
| `boost_underpred4_p004` | `wall,sky,person,road=0.04` | 4.453 | +0.391 |
| `underpred_p002_spurious_m002` | `wall,sky,person,road=0.02`; `house,screen door,bookcase,skyscraper,swivel chair=-0.02` | 4.404 | +0.342 |
| `boost_frequent7_p002` | `wall,building,sky,floor,tree,person,road=0.02` | 4.391 | +0.329 |
| no-bias recovered baseline | none | 4.061 | 0.000 |

## Full ADE20K confirmation

Full summary: `outputs/diagnostics/ade20k_group_calibration_full/summary.json`.

| Full probe | Class bias | ADE20K mIoU | Delta vs recovered no-bias full |
|---|---|---:|---:|
| `boost_frequent7_p004` | `wall,building,sky,floor,tree,person,road=0.04` | 11.470 | +0.920 |
| `boost_underpred4_p004` | `wall,sky,person,road=0.04` | 11.219 | +0.670 |
| `underpred_p002_spurious_m002` | `wall,sky,person,road=0.02`; `house,screen door,bookcase,skyscraper,swivel chair=-0.02` | 11.022 | +0.473 |
| recovered no-bias full | none | 10.549 | 0.000 |

Best full row frequent-class behavior:

| Class | IoU | pred/target | Recall | Precision |
|---|---:|---:|---:|---:|
| wall | 28.56 | 0.721 | 0.382 | 0.530 |
| building | 35.56 | 0.869 | 0.490 | 0.564 |
| sky | 17.25 | 0.205 | 0.177 | 0.864 |
| floor | 28.34 | 1.750 | 0.607 | 0.347 |
| tree | 39.37 | 0.946 | 0.550 | 0.581 |
| person | 25.81 | 0.724 | 0.354 | 0.489 |
| road | 43.26 | 1.149 | 0.649 | 0.565 |

The frequent-class boost improves the main underprediction failure mode (`wall/building/sky/tree/person/road`) but overuses floor. Despite this, the 150-class mean improves on the full validation split.

## Diagnostic selected segmentation aggregate

Machine summary: `outputs/ablations/segmentation_full_selected/summary_ade20k_group_calibrated_diagnostic.json`.

This aggregate keeps the previous selected VOC20 / Context rows because the `last_hidden_state` sanity check hurt VOC20 and slightly hurt Context, then plugs in the best ADE20K targeted-calibrated row.

| Dataset | mIoU | Source |
|---|---:|---|
| VOC20 | 37.573 | previous selected explicit hidden-state row |
| Pascal Context | 22.004 | previous selected explicit hidden-state row |
| ADE20K | 11.470 | recovered `last_hidden_state` + clean aliases + `--ignore-zero` + frequent7 `+0.04` bias |
| Average | 23.682 | diagnostic aggregate |

Paper average is `23.867`; the diagnostic calibrated aggregate is `99.23%` of the paper average, with gap `-0.184`.

## Decision / caveat

- Do **not** switch VOC20 / Context to `last_hidden_state`: it drops VOC20 from `37.57` to `33.57` and Context from `22.00` to `21.90`.
- Targeted ADE20K class/group bias is promising: full ADE20K improves `10.55 -> 11.47`.
- This calibration is validation-informed because the group and magnitude were selected from ADE20K diagnostics. Treat the `23.68` selected average as a **diagnostic calibrated row**, not an unqualified paper-protocol row, unless a separate calibration protocol is accepted.
- If stricter calibration evidence is needed next, split ADE20K val into a small calibration subset and a held-out evaluation subset, or derive class groups/biases from train metadata without looking at validation predictions.
