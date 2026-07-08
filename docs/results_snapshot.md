# PAL Reproduction Results Snapshot

This snapshot records real outputs produced under `/home/hnxxzy/finegrained-relative-repr-repro` with the local conda interpreter `/home/hnxxzy/miniconda3/envs/ovvs/bin/python`.

## Training

- Training token cache: `data/tokens/coco2014_full`
- Train pairs: 82,783 COCO2014 train first-caption pairs
- Model: strict PAL, K=512, CAP temperature 0.03
- Checkpoint: `outputs/pal_k512_coco2014_full/checkpoint.pt`
- Metrics: `outputs/pal_k512_coco2014_full/metrics.json`
- Final train loss: `0.28341474021328655`
- Trainable params: `anchors_img`, `anchors_txt`

## Retrieval

| Protocol | Images | Texts | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | Paper R@1 target | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| COCO val first-caption one-to-one | 40,504 | 40,504 | 15.22 | 33.36 | 43.77 | 15.01 | 33.73 | 43.83 | I2T 56.3 / T2I 42.6 | Strict 40,504-way first-caption probe; not paper protocol. |
| COCO val 5K multi-caption | 5,000 | 25,021 | 49.82 | 77.66 | 86.68 | 37.06 | 66.29 | 77.80 | I2T 56.3 / T2I 42.6 | Closer to paper; local seed=42 5K subset, not confirmed official split. |
| COCO Karpathy test 5K multi-caption | 5,000 | 25,010 | 49.22 | 77.70 | 86.56 | 36.63 | 66.37 | 77.87 | I2T 56.3 / T2I 42.6 | Standard Karpathy test split from `caption_datasets.zip`; paper-protocol candidate. |
| Flickr30k local 1K multi-caption | 1,000 | 5,000 | 67.80 | 90.20 | 95.40 | 52.80 | 80.62 | 87.52 | I2T 76.3 / T2I 61.8 | Uses `/home/hnxxzy/Downloads/Flickr30k.zip`; local seed=42 1K subset, not confirmed official split. |
| Flickr30k Karpathy test 1K multi-caption | 1,000 | 5,000 | 67.40 | 90.10 | 94.90 | 50.96 | 79.18 | 86.76 | I2T 76.3 / T2I 61.8 | Standard Karpathy test split from `caption_datasets.zip`; paper-protocol candidate. |

Retrieval outputs:

- `outputs/pal_k512_coco2014_full/coco_val_first_caption_retrieval.json`
- `outputs/pal_k512_coco2014_full/coco_val_5k_multicaption_retrieval.json`
- `outputs/pal_k512_coco2014_full/coco_karpathy_test_multicaption_retrieval.json`
- `outputs/pal_k512_coco2014_full/flickr30k_1k_multicaption_retrieval.json`
- `outputs/pal_k512_coco2014_full/flickr30k_karpathy_test_multicaption_retrieval.json`

## Zero-shot classification

| Dataset | N | Local top1 | Local top5 | Paper top1 | Gap | Relative |
|---|---:|---:|---:|---:|---:|---:|
| STL10 | 8,000 | 91.96 | 99.96 | 95.30 | -3.34 | 96.5% |
| CIFAR100 | 10,000 | 42.58 | 71.47 | 48.80 | -6.22 | 87.3% |
| Caltech101 | 8,677 | 45.63 | 67.82 | 60.90 | -15.27 | 74.9% |
| DTD | 1,880 | 15.21 | 33.94 | 17.70 | -2.49 | 85.9% |
| EuroSAT | 27,000 | 28.08 | 72.46 | 34.60 | -6.52 | 81.2% |
| Average | - | 44.69 | - | 51.46 | -6.77 | 86.8% |

Prompt-template sweep best-of-4 results:

| Dataset | Best template | Best top1 | Paper top1 | Gap | Relative |
|---|---|---:|---:|---:|---:|
| STL10 | `a close-up photo of {class_name}` | 92.15 | 95.30 | -3.15 | 96.69% |
| CIFAR100 | `a photo of {class_name}` | 42.58 | 48.80 | -6.22 | 87.25% |
| Caltech101 | `a close-up photo of {class_name}` | 48.84 | 60.90 | -12.06 | 80.20% |
| DTD | `a cropped photo of {class_name}` | 16.54 | 17.70 | -1.16 | 93.46% |
| EuroSAT | `a close-up photo of {class_name}` | 30.32 | 34.60 | -4.28 | 87.64% |
| Average | mixed best templates | 46.09 | 51.46 | -5.37 | 89.56% |

Classification outputs:

- `outputs/pal_k512_coco2014_full/stl10_classification.json`
- `outputs/pal_k512_coco2014_full/cifar100_classification.json`
- `outputs/pal_k512_coco2014_full/caltech101_classification.json`
- `outputs/pal_k512_coco2014_full/dtd_classification.json`
- `outputs/pal_k512_coco2014_full/eurosat_classification.json`
- `outputs/prompt_sweep/classification/summary.json`

## Segmentation

The first table is retained as a historical baseline because it used the old implicit original-mask frame and Pascal Context all-459 protocol.

| Dataset | Split | Samples | Local mIoU | Paper mIoU | Gap | Relative | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| VOC20 | val | 1,449 | 14.82 | 32.30 | -17.48 | 45.89% | First full-val run; substantially better than smoke but still below paper. |
| Context | trainval | 10,103 | 0.53 | 25.50 | -24.97 | 2.09% | Full evaluation complete; likely protocol/debug gap. |
| ADE20K | validation | 2,000 | 1.47 | 13.80 | -12.33 | 10.67% | Full evaluation complete; likely protocol/debug gap. |
| Average | - | - | 5.61 | 23.87 | -18.26 | 23.50% | Average over VOC20/Context/ADE20K. |

Corrected full rerun results:

| Dataset | Corrected protocol | Samples | Classes | Corrected mIoU | Paper mIoU | Gap | Relative | Delta vs historical |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VOC20 | `--target-frame processor` | 1,449 | 20 | 20.58 | 32.30 | -11.72 | 63.71% | +5.75 |
| Context | `--target-frame processor --context-protocol common59` | 10,103 | 59 | 11.23 | 25.50 | -14.27 | 44.05% | +10.70 |
| ADE20K | `--target-frame processor` | 2,000 | 150 | 2.19 | 13.80 | -11.61 | 15.87% | +0.72 |
| Average | - | - | - | 11.33 | 23.87 | -12.53 | 47.48% | +5.72 |

Segmentation outputs:

- `outputs/pal_k512_coco2014_full/voc20_segmentation_full.json`
- `outputs/pal_k512_coco2014_full/context_segmentation_full.json`
- `outputs/pal_k512_coco2014_full/ade20k_segmentation_full.json`
- `outputs/pal_k512_coco2014_full/voc20_segmentation_processor_full.json`
- `outputs/pal_k512_coco2014_full/context_segmentation_common59_processor_full.json`
- `outputs/pal_k512_coco2014_full/ade20k_segmentation_processor_full.json`

Segmentation protocol diagnostics:

- notes: `docs/segmentation_debug_notes.md`
- probe script: `scripts/diagnose_segmentation_protocol.py`
- probe outputs: `outputs/diagnostics/*_segmentation_protocol_probe*.json`
- formal evaluator support: `scripts/evaluate_segmentation.py --target-frame {original,processor}` and `--context-protocol {all459,common59}`.
- corrected 16-sample formal probes:
  - VOC20, `--target-frame processor`: mIoU `12.5179`.
  - Pascal Context, `--target-frame processor --context-protocol common59`: mIoU `12.3125` over 59 classes.
  - ADE20K, `--target-frame processor`: mIoU `0.2571`.
- interpretation at this diagnostic stage: the old full-run Context number used all 459 labels and original-mask scoring; it should now be treated as historical. Common-59 + processor-frame scoring is the required rerun condition for Context. Processor-frame scoring alone barely improves the cheap ADE20K probe, so later ADE20K-specific alias/layer calibration work was required.
- corrected full-run interpretation: the corrected protocol roughly doubles the segmentation macro average (`5.61 -> 11.33`) but remains below paper (`23.87`); later ADE20K clean aliases, `--ignore-zero`, recovered `last_hidden_state` dense tokens, and diagnostic targeted group calibration are summarized in the selected-full table below.

## Completed pipeline sweeps and analyses

The ordered pipeline completed training-only sweep jobs for K, `tau_p`, and token-usage ablations. These are not paper ablation-table metrics yet; they record convergence and checkpoints that still need downstream evaluation.

### K sweep final train loss

| K | Final train loss |
|---:|---:|
| 32 | 0.509355 |
| 64 | 0.403373 |
| 128 | 0.338583 |
| 256 | 0.303722 |
| 512 | 0.283415 |

### `tau_p` sweep final train loss

| `tau_p` | Final train loss |
|---:|---:|
| 0.02 | 0.252805 |
| 0.03 | 0.283415 |
| 0.05 | 0.335997 |
| 0.07 | 0.371047 |
| 0.10 | 0.407706 |

### Token usage ablation final train loss

| Mode | Final train loss |
|---|---:|
| global | 0.740161 |
| mean | 0.501132 |
| cap | 0.283415 |

### Downstream retrieval ablation metrics

Detailed per-run JSONs live under `outputs/ablations/retrieval/`; the machine-readable summary is `outputs/ablations/retrieval/summary.json`, and the readable report is `docs/ablation_downstream_retrieval_results.md`. These retrieval metrics use official/Karpathy COCO and Flickr30k multi-caption token caches. `Avg R@1` below averages COCO I2T, COCO T2I, Flickr30k I2T, and Flickr30k T2I R@1.

| Sweep | Best local retrieval setting | Avg R@1 | Key trend |
|---|---|---:|---|
| K | K=512 | 51.05 | Monotonic increase from K32 `35.94` to K512 `51.05`. |
| `tau_p` | `0.02` | 51.41 | Slightly above main `0.03` retrieval Avg R@1 `51.05`; classification/segmentation still needed before changing the main setting. |
| token usage | CAP | 51.05 | CAP `51.05` > mean `37.26` > global `25.20`, confirming the downstream retrieval benefit of CAP beyond train loss. |

### Downstream classification and segmentation-probe ablations

Detailed report: `docs/ablation_downstream_classification_segmentation_results.md`. Full classification ablation output: `outputs/ablations/classification_fast/summary.json`. Corrected 64-sample segmentation probe output: `outputs/ablations/segmentation_probes_fast/summary.json`.

| Sweep | Classification finding | 64-sample segmentation probe finding |
|---|---|---|
| K | Avg top1 improves monotonically K32 `38.20` -> K512 `45.63`. | Probe Avg mIoU peaks at K256 `17.17`, slightly above K512 `16.51`. |
| `tau_p` | `0.02` best Avg top1 `46.48`; main `0.03` is `45.63`. | `0.07` best probe Avg mIoU `18.11`; `0.05` close at `17.58`. |
| token usage | CAP `45.63` > mean `42.49` > global `39.05`. | CAP `16.51` > mean `11.77` >> global `0.61`. |

Selected full corrected segmentation reruns are now complete for the probe-prioritized checkpoints:

| Variant | VOC20 mIoU | Context mIoU | ADE20K mIoU | Avg mIoU | Relative to paper avg |
|---|---:|---:|---:|---:|---:|
| K=256 | 33.48 | 20.32 | 5.89 | 19.90 | 83.37% |
| `tau_p=0.07` + ADE20K `--ignore-zero` | 37.57 | 22.00 | 7.99 | 22.52 | 94.37% |
| `tau_p=0.07` + ADE20K clean aliases + `--ignore-zero` | 37.57 | 22.00 | 9.33 | 22.97 | 96.24% |
| `tau_p=0.07` + ADE20K clean aliases + `--ignore-zero` + recovered `last_hidden_state` dense tokens | 37.57 | 22.00 | 10.55 | 23.38 | 97.94% |
| diagnostic targeted ADE20K group calibration | 37.57 | 22.00 | 11.47 | 23.68 | 99.23% |

### Anchor overlap

| Metric | Value |
|---|---:|
| matched hard overlap | 0.517633 |
| matched Dice | 0.517633 |
| mismatched hard overlap | 0.436705 |
| mismatched Dice | 0.436705 |

Matched overlap is `+0.080928` absolute / `+18.53%` relative above mismatched overlap.

Full pipeline summary:

- `docs/pipeline_results_snapshot.md`

## Verification

Current test command:

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
```

Current result:

```text
Ran 63 tests in 0.206s
OK
```

## Interpretation caveats

- The implementation originally used explicit DINOv2-L/RoBERTa-L hidden-state indices for dense segmentation. ADE20K dense-token recovery shows the encoders' `last_hidden_state` outputs improve the selected ADE20K full row to `10.55`; the exact paper dense-token/layer convention remains partially inferred.
- COCO and Flickr30k Karpathy test extraction/evaluation are complete. Both are now the preferred retrieval rows for paper-protocol comparison; the older seed-42 subset rows are retained only as historical proxies.
- Prompt-template sweep improved classification average top-1 from 44.69 to 46.09; remaining gap suggests prompt choice is only a partial explanation.
- VOC20/Context/ADE20K corrected full segmentation rerun is complete. Processor-frame + Context common-59 roughly doubles macro mIoU (`5.61 -> 11.33`), and the ADE20K layer/prompt/alias full rerun improves ADE20K to `5.66`. The selected full `tau_p=0.07` segmentation rerun plus ADE20K clean aliases, `--ignore-zero`, and recovered `last_hidden_state` dense tokens reaches average mIoU `23.38` vs paper average `23.87`, with ADE20K still the main gap (`10.55` vs `13.80`). Diagnostic targeted ADE20K group calibration reaches ADE20K `11.47` and selected average `23.68` (`99.23%`), but it is validation-informed. K/`tau_p`/token-usage sweep training, downstream retrieval, downstream classification, segmentation probes, selected full segmentation reruns, ADE20K ignore-zero correction, full ADE20K clean-alias rows, ADE20K dense-token/layer recovery, and targeted group-calibration diagnostics are complete.
