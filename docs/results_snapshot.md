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
- interpretation: the old full-run Context number used all 459 labels and original-mask scoring; it should now be treated as historical. Common-59 + processor-frame scoring is the required rerun condition for Context. ADE20K remains unresolved because processor-frame scoring alone barely improves the cheap probe.
- corrected full-run interpretation: the corrected protocol roughly doubles the segmentation macro average (`5.61 -> 11.33`) but remains below paper (`23.87`); ADE20K is still the main unresolved dense-evaluation gap.

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
Ran 47 tests in 0.183s
OK
```

## Interpretation caveats

- The implementation uses final DINOv2-L/RoBERTa-L hidden tokens. The paper mentions CKA-based layer selection but the exact layer indices were not recovered from the PDF text.
- COCO and Flickr30k Karpathy test extraction/evaluation are complete. Both are now the preferred retrieval rows for paper-protocol comparison; the older seed-42 subset rows are retained only as historical proxies.
- Prompt-template sweep improved classification average top-1 from 44.69 to 46.09; remaining gap suggests prompt choice is only a partial explanation.
- VOC20/Context/ADE20K corrected full segmentation rerun is complete. Processor-frame + Context common-59 roughly doubles macro mIoU (`5.61 -> 11.33`), but still trails the paper average (`23.87`), with ADE20K remaining the main unresolved dense-evaluation gap. K/`tau_p`/token-usage sweep training is complete; downstream ablation metrics remain to be run.
