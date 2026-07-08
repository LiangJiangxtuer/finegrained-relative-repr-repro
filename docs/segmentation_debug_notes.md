# Segmentation Protocol Debug Notes

Updated: 2026-07-05 04:38 EDT

This note records the systematic-debugging passes for the low VOC20 / Pascal Context / ADE20K segmentation numbers.

## Symptom

Full segmentation evaluation now runs end-to-end, but the metrics are far below the paper:

| Dataset | Full-run mIoU-fg | Paper | Relative |
|---|---:|---:|---:|
| VOC20 | 14.82 | 32.30 | 45.89% |
| Pascal Context | 0.53 | 25.50 | 2.09% |
| ADE20K | 1.47 | 13.80 | 10.67% |

The failure is not a crash; it is a protocol/quality failure in dense zero-shot evaluation.

## Diagnostic loop added

A lightweight probe script was added:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python \
  scripts/diagnose_segmentation_protocol.py \
  --dataset <VOC20|Context|ADE20K> \
  --checkpoint outputs/pal_k512_coco2014_full/checkpoint.pt \
  --output outputs/diagnostics/<dataset>_segmentation_protocol_probe.json \
  --limit 16 \
  --batch-size 4 \
  --local-files-only \
  --also-ignore-zero
```

The probe compares two target frames:

1. `original_target`: existing full-run behavior, upsampling patch logits to the original mask shape.
2. `processor_aligned_target`: transforms the target mask with the same DINOv2 HF image-processor geometry before scoring.

The local DINOv2 processor does:

```text
do_resize=True, shortest_edge=256
do_center_crop=True, crop_size=224x224
```

Therefore dense predictions are made on a 224x224 center-cropped image frame. Comparing those predictions directly to the uncropped original mask is geometrically inconsistent.

## Probe results on first 16 samples

| Dataset | Frame / option | mIoU-fg |
|---|---|---:|
| VOC20 | original target, ignore 255 | 9.18 |
| VOC20 | processor-aligned target, ignore 255 | 12.52 |
| VOC20 | processor-aligned target, ignore 0 | 15.01 |
| Context | original target | 0.53 |
| Context | processor-aligned target | 0.76 |
| ADE20K | original target | 0.216 |
| ADE20K | processor-aligned target | 0.257 |
| ADE20K | processor-aligned target, ignore 0 | 0.257 |

Outputs:

- `outputs/diagnostics/voc20_segmentation_protocol_probe.json`
- `outputs/diagnostics/context_segmentation_protocol_probe.json`
- `outputs/diagnostics/ade20k_segmentation_protocol_probe.json`

## Formal protocol update

The geometry/class-protocol fixes were promoted from the diagnostic script into the formal segmentation evaluator:

- `scripts/evaluate_segmentation.py --target-frame {original,processor}`
  - `original` preserves the historical behavior for traceability.
  - `processor` scores against masks transformed by the same DINOv2 image-processor resize + center-crop used for patch extraction.
- `scripts/evaluate_segmentation.py --context-protocol {all459,common59}`
  - `all459` preserves the original Pascal Context 459-label path.
  - `common59` evaluates the common 59-class Pascal Context protocol while keeping the raw dataset label IDs.
- `src/pal_repro/segmentation.py` now exposes shared helpers for processor-frame mask transforms, Context common-59 label selection, prompt-friendly class-name normalization, and non-contiguous label-ID mapping.

The formal evaluator probe outputs from the corrected path are:

| Dataset | Formal evaluator options | Samples | Classes | mIoU-fg |
|---|---|---:|---:|---:|
| VOC20 | `--target-frame processor` | 16 | 20 | 12.52 |
| Context | `--target-frame processor --context-protocol common59` | 16 | 59 | 12.31 |
| ADE20K | `--target-frame processor` | 16 | 150 | 0.257 |

Outputs:

- `outputs/diagnostics/voc20_processor_segmentation_probe_eval.json`
- `outputs/diagnostics/context_common59_processor_segmentation_probe_eval.json`
- `outputs/diagnostics/ade20k_processor_segmentation_probe_eval.json`
- `outputs/diagnostics/context_common59_segmentation_protocol_probe.json`

The important new result is Context: moving from the previous all-459, processor-aligned probe (`0.76`) to common-59, processor-aligned scoring gives `12.31` mIoU on the same 16-sample probe. This strongly supports the hypothesis that the original Context number was dominated by protocol mismatch, not just model failure.

## Corrected full rerun results

The corrected full segmentation rerun completed normally on 2026-07-05:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python scripts/evaluate_segmentation.py \
  --dataset VOC20 \
  --checkpoint outputs/pal_k512_coco2014_full/checkpoint.pt \
  --output outputs/pal_k512_coco2014_full/voc20_segmentation_processor_full.json \
  --target-frame processor \
  --batch-size 8 \
  --local-files-only

PYTHONUNBUFFERED=1 PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python scripts/evaluate_segmentation.py \
  --dataset Context \
  --checkpoint outputs/pal_k512_coco2014_full/checkpoint.pt \
  --output outputs/pal_k512_coco2014_full/context_segmentation_common59_processor_full.json \
  --target-frame processor \
  --context-protocol common59 \
  --batch-size 8 \
  --local-files-only

PYTHONUNBUFFERED=1 PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python scripts/evaluate_segmentation.py \
  --dataset ADE20K \
  --checkpoint outputs/pal_k512_coco2014_full/checkpoint.pt \
  --output outputs/pal_k512_coco2014_full/ade20k_segmentation_processor_full.json \
  --target-frame processor \
  --batch-size 8 \
  --local-files-only
```

| Dataset | Corrected protocol | Samples | Classes | Old full mIoU | Corrected mIoU | Paper | Delta vs old | Relative to paper |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VOC20 | processor frame | 1,449 | 20 | 14.82 | 20.58 | 32.30 | +5.75 | 63.71% |
| Pascal Context | processor frame + common59 | 10,103 | 59 | 0.53 | 11.23 | 25.50 | +10.70 | 44.05% |
| ADE20K | processor frame | 2,000 | 150 | 1.47 | 2.19 | 13.80 | +0.72 | 15.87% |
| Average | - | - | - | 5.61 | 11.33 | 23.87 | +5.72 | 47.48% |

Outputs:

- `outputs/pal_k512_coco2014_full/voc20_segmentation_processor_full.json`
- `outputs/pal_k512_coco2014_full/context_segmentation_common59_processor_full.json`
- `outputs/pal_k512_coco2014_full/ade20k_segmentation_processor_full.json`
- log: `outputs/logs/corrected_segmentation_rerun_20260705_104725.log`

Interpretation: the corrected protocol roughly doubles the macro segmentation average (`5.61 -> 11.33`) and substantially improves VOC20/Context, but it is still below the paper average (`23.87`). ADE20K remains the biggest unresolved dense-evaluation gap after protocol correction.

## What this falsifies / supports

### H1: mask/image geometry mismatch contributes to the low metric

Supported, but not sufficient. Scoring against processor-aligned masks improves the 16-sample VOC20/Context/ADE probes, but the gain is too small to explain the entire gap to paper.

### H2: ADE20K/Context low scores are mainly caused by ignore-index handling

Mostly falsified for the tested subset. ADE20K `ignore 0` changes little (`0.25706 -> 0.25718` on aligned target), and Context masks in the first 16 examples do not expose a meaningful zero-label issue.

### H3: prompt template alone fixes the issue

Falsified for the cheap probe. Switching from `a photo of {class_name}` to `a {class_name}` made Context and ADE20K worse on the first 16 samples.

Outputs:

- `outputs/diagnostics/context_segmentation_protocol_probe_a_object.json`
- `outputs/diagnostics/ade20k_segmentation_protocol_probe_a_object.json`

### H4: class protocol / class-name mapping is a major remaining issue

Supported by evidence.

- The initial Pascal Context path evaluated all 459 labels from `labels.txt`. The new common-59 path produces a large 16-sample gain (`0.76 -> 12.31` in processor frame), so Context protocol mismatch is a confirmed major contributor.
- Context predicted labels are concentrated on classes such as `hard disk drive`, `photo`, `eyeglass`, while target pixels in the probe are dominated by `sky`, `tvmonitor`, `person`, `building`, `ground`, etc.
- ADE20K predictions are concentrated on labels such as `ottoman`, `awning`, `stairway`, `chandelier`, while target pixels are dominated by `building`, `wall`, `sky`, `floor`, `ceiling`, etc.

This points beyond a pure mIoU accumulator bug: the dense class logits themselves are poorly calibrated for frequent scene classes.

## Root-cause ranking after corrected-protocol rerun

1. **Context protocol mismatch was a major bug.** The evaluator now supports `--context-protocol common59`; full Context improves from `0.53` to `11.23` mIoU.
2. **Dense evaluation geometry was implicit and partially wrong.** The evaluator now supports `--target-frame processor`; old full metrics remain historical, while corrected full VOC20 improves from `14.82` to `20.58`.
3. **ADE20K was the remaining dataset-specific bottleneck.** Processor-frame scoring alone improved full ADE20K only from `1.47` to `2.19`, but later clean aliases, `--ignore-zero`, recovered `last_hidden_state` dense tokens, and targeted group calibration raised diagnostic ADE20K to `11.47`.
4. **Class-name normalization / prompt engineering mattered for ADE20K.** Conservative clean aliases avoid misleading WordNet terms and improve the selected ADE20K row from `7.99` to `9.33`.
5. **Layer-selection / dense-token choice was mismatched for ADE20K.** Recovered `last_hidden_state` dense tokens improve ADE20K to `10.55`, but VOC20/Context sanity shows this protocol should not be applied globally.

## Next recommended actions

1. Treat the previous full-run segmentation JSON files as historical/baseline, because they used the old implicit original-mask frame and all-459 Context path.
2. Treat targeted ADE20K group calibration as validation-informed diagnostics unless a held-out calibration protocol is added.
3. If Context needs further improvement beyond `22.00` selected full mIoU, the next debug layer is Context-specific dense-token layer selection and prompt ensemble, not a global ADE20K `last_hidden_state` switch.
4. Future segmentation tasks should keep explicit protocol flags in output names/JSON: target frame, Context protocol, alias policy, `--ignore-zero`, dense-token layer choice, and any diagnostic class bias.

## ADE20K follow-up after selected full `tau_p=0.07`

Updated on 2026-07-06 after the selected full `tau_p=0.07` segmentation rerun made ADE20K the remaining bottleneck. Detailed report: `docs/ade20k_dense_debug_results.md`.

Key findings:

- Image-size probes at 224/336/448 did not improve ADE20K limit-64 mIoU (`2.83`, `2.86`, `2.75`).
- Prompt/layer probes were noisy: `a photo of a {class_name}` with `text_layer=-4` looked best on limit 64 (`3.22`) but underperformed on the full split (`6.96` without ignore-zero, `7.50` with ignore-zero).
- ADE20K annotations include label id 0 void/unlabeled pixels. Across 2,000 validation masks, zero-label pixels average `8.39%`; 430 images have more than 10% zero-label pixels.
- `scripts/evaluate_segmentation.py --ignore-zero` was added to treat label id 0 as void during mIoU accumulation.
- Full ADE20K selected `tau_p=0.07` improves from `7.62` to `7.99` with `--ignore-zero` using the original best 4-template ensemble and `vision_layer=-1`, `text_layer=-2`.
- Updated selected full summary: `outputs/ablations/segmentation_full_selected/summary_ade20k_ignore0.json`; average segmentation mIoU becomes `22.52` vs paper `23.87`.

## ADE20K alias cleanup and calibration follow-up

Updated on 2026-07-07. Detailed generated report: `docs/ade20k_frequent_class_error_analysis.md`.

- `src/pal_repro/segmentation.py` now contains a conservative ADE20K clean-alias table that avoids broad WordNet aliases such as `route`, `mortal/soul`, `machine`, `mantle/pall`, `throne/stool`, `wheel/cycle`, and `idiot box`.
- The ordered pipeline now runs ADE20K segmentation with `--alias-policy clean --ignore-zero` so future reruns avoid misleading raw aliases by default.
- The completed full clean-alias ADE20K all-variant run is `outputs/ablations/segmentation_full_ade20k_clean_ignore0_all_variants/summary.json`; best row is `tau_p=0.07` with ADE20K mIoU `9.33`, improving the selected full average to `22.97` (`96.24%` of the paper average target) when paired with the existing VOC20 `37.57` and Context `22.00` rows.
- Frequent-class probe analysis over `wall/building/sky/floor/tree/person/road` shows that uncalibrated alias policies remain best on mean mIoU (`alias_first` 3.069, `alias_all` 3.032, `alias_clean` 2.962 on limit 64 with `--ignore-zero`).
- Simple class-prior logit bias is not recommended for ADE20K: ADE20K-ratio prior alpha=0.1 drops to `2.43`, alpha=0.25 collapses to `0.54`, and train-count alpha=0.25 reaches only `0.97`; foreground/background calibration is not indicated beyond treating label id 0 as void.
- Dense-token/layer protocol recovery then found that using encoder `last_hidden_state` outputs for ADE20K dense scoring improves the selected full ADE20K row to `10.55`; updated selected average is `23.38` (`97.94%` of paper average). Diagnostic image-class center/zscore calibration hurts the 64-sample loop and should not be promoted. See `docs/ade20k_dense_protocol_recovery.md`.
- VOC20/Context sanity check shows `last_hidden_state` is ADE20K-specific: VOC20 `37.57 -> 33.57`, Context `22.00 -> 21.90`. Targeted ADE20K group calibration with `wall,building,sky,floor,tree,person,road=0.04` improves full ADE20K `10.55 -> 11.47`; selected diagnostic average is `23.68` (`99.23%`) but should be labeled validation-informed. See `docs/ade20k_group_calibration_results.md`.
- Foreground/background calibration is not indicated for ADE20K beyond ignoring label id 0 as void; adding a predicted background class would be a separate VOC/Context protocol experiment rather than an ADE20K fix.
