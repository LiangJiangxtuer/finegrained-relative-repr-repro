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

Classification outputs:

- `outputs/pal_k512_coco2014_full/stl10_classification.json`
- `outputs/pal_k512_coco2014_full/cifar100_classification.json`
- `outputs/pal_k512_coco2014_full/caltech101_classification.json`
- `outputs/pal_k512_coco2014_full/dtd_classification.json`
- `outputs/pal_k512_coco2014_full/eurosat_classification.json`

## Verification

Current test command:

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
```

Current result:

```text
Ran 42 tests in 1.371s
OK
```

## Interpretation caveats

- The implementation uses final DINOv2-L/RoBERTa-L hidden tokens. The paper mentions CKA-based layer selection but the exact layer indices were not recovered from the PDF text.
- COCO and Flickr30k Karpathy test extraction/evaluation are complete. Both are now the preferred retrieval rows for paper-protocol comparison; the older seed-42 subset rows are retained only as historical proxies.
- Prompting uses `a photo of {class_name}`. Paper prompt templates were not explicit in the PDF text, so prompt/template tuning may explain part of the remaining classification gap.
- Segmentation and ablation experiments remain incomplete.
