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

## Current root-cause ranking

1. **Context protocol mismatch was a major bug.** The evaluator now supports `--context-protocol common59`; full Context improves from `0.53` to `11.23` mIoU.
2. **Dense evaluation geometry was implicit and partially wrong.** The evaluator now supports `--target-frame processor`; old full metrics remain historical, while corrected full VOC20 improves from `14.82` to `20.58`.
3. **ADE20K remains unresolved.** Processor-frame scoring improves full ADE20K only from `1.47` to `2.19`, so the remaining gap likely needs dataset-specific prompt/name cleanup, dense-layer selection, or another paper-protocol detail.
4. **Class-name normalization / prompt engineering is still incomplete.** Minimal aliases (`tvmonitor -> tv monitor`, `pottedplant -> potted plant`, `bedclothes -> bed clothes`) are implemented, but ADE20K aliases and prompt ensembles are not.
5. **Layer-selection / dense-token choice may be mismatched.** Retrieval/classification are much closer to paper than segmentation, so dense patch alignment may be particularly sensitive to the paper's CKA-selected layers.

## Next recommended actions

1. Treat the previous full-run segmentation JSON files as historical/baseline, because they used the old implicit original-mask frame and all-459 Context path.
2. Before spending more full ADE20K time, run 16-64 sample ADE20K prompt/name probes; the geometry fix alone did not help enough.
3. If Context needs further improvement beyond `11.23`, the next debug layer is dense-token layer selection and prompt ensemble, not mIoU bookkeeping.
4. Update the ordered pipeline so future segmentation tasks use corrected output names and explicit protocol flags.
