# PAL Full Reproduction Handoff

Use this file if the chat context is compacted or a new session is opened.

## Active repository

`/home/hnxxzy/finegrained-relative-repr-repro`

Use Python:

`/home/hnxxzy/miniconda3/envs/ovvs/bin/python`

Always set `PYTHONPATH=src` for module commands.

## Completed core reproduction steps

1. Paper text extracted to `docs/paper_text.txt`.
2. Reference implementation cloned to `external/bridge-anchors`.
3. Paper claims/formulas/experiment matrix written:
   - `docs/paper_claims_and_experiments.md`
   - `docs/full_reproduction_plan.md`
   - `configs/reproduction_matrix.yaml`
4. Strict PAL core implemented with anchor-only trainable params:
   - `src/pal_repro/models/pal.py`
   - `src/pal_repro/losses.py`
   - `src/pal_repro/train.py`
   - `src/pal_repro/eval.py`
   - `src/pal_repro/evaluate.py`
5. COCO2014 train token extraction completed and merged:
   - token dir: `data/tokens/coco2014_full`
   - `image_tokens.pt`: `(82783, 257, 1024)` fp16, ~40.58 GiB
   - `text_tokens.pt`: `(82783, 64, 1024)` fp16, ~10.11 GiB
   - `text_mask.pt`: `(82783, 64)` bool
   - `metadata_merged.json` format: `monolithic_from_chunks`
6. K=512 full PAL training completed:
   - output dir: `outputs/pal_k512_coco2014_full`
   - checkpoint: `outputs/pal_k512_coco2014_full/checkpoint.pt`
   - final train loss: `0.28341474021328655`
   - train samples: 82,783
   - eval split: 0 because the run used all training samples
   - trainable params in checkpoint: `anchors_img`, `anchors_txt`
7. COCO retrieval completed:
   - first-caption output: `outputs/pal_k512_coco2014_full/coco_val_first_caption_retrieval.json`
   - first-caption I2T/T2I R@1: `15.22 / 15.01`
   - 5K multi-caption output: `outputs/pal_k512_coco2014_full/coco_val_5k_multicaption_retrieval.json`
   - 5K multi-caption images/texts: `5000 / 25021`
   - 5K multi-caption I2T/T2I R@1: `49.82 / 37.06`
   - Flickr30k 1K multi-caption output: `outputs/pal_k512_coco2014_full/flickr30k_1k_multicaption_retrieval.json`
   - Flickr30k 1K multi-caption images/texts: `1000 / 5000`
   - Flickr30k I2T/T2I R@1: `67.80 / 52.80`
   - paper Flickr30k R@1 target: `76.3 / 61.8`
   - paper COCO R@1 target: `56.3 / 42.6`
8. Added multi-caption protocol support:
   - `scripts/extract_coco_tokens.py --caption-policy all`
   - `scripts/extract_flickr30k_tokens.py` reads `/home/hnxxzy/Downloads/Flickr30k.zip` directly without unpacking
   - `pal_repro.eval.multicaption_retrieval_metrics`
   - `python -m pal_repro.evaluate retrieval-multicaption ...`
9. Zero-shot classification completed:
   - STL10 top1: `91.96` vs paper `95.30`
   - CIFAR100 top1: `42.58` vs paper `48.80`
   - Caltech101 top1: `45.63` vs paper `60.90`
   - DTD top1: `15.21` vs paper `17.70`
   - EuroSAT top1: `28.08` vs paper `34.60`
   - average top1: `44.69` vs paper `51.46`
10. Results snapshot written to `docs/results_snapshot.md`.
11. Tests currently pass:
   - command: `PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v`
   - result: `Ran 26 tests ... OK`

## Current active long-running process

None at this handoff point.

## Next commands

Segmentation and ablations are the next missing experiment groups. For paper-grade parity, implement/run:

```text
VOC20 / Pascal Context / ADE20K dense segmentation evaluation
anchor_count ablations: K in [32, 64, 128, 256, 512]
pool_temperature ablations: tau_p in [0.02, 0.03, 0.05, 0.07, 0.1]
```

Classification and COCO retrieval outputs are already under `outputs/pal_k512_coco2014_full/`.

## Known gaps for paper-level parity

1. Flickr30k is now available and evaluated from `/home/hnxxzy/Downloads/Flickr30k.zip`; exact official split still needs to be matched if required for strict parity.
2. Exact paper retrieval protocol likely needs standard COCO/Flickr official/Karpathy splits; current COCO/Flickr runs use deterministic local seed-42 subsets.
3. Zero-shot classification runners/results are complete, but remaining gap analysis should investigate prompt templates, exact encoder layer selection, and Caltech101 split/protocol differences.
4. Segmentation data exists locally for VOC20/Context/ADE20K, but dense token-to-class segmentation runner still needs to be completed.
5. Ablations for anchor count and pool temperature remain to run.

## Important caveat

Conda may print plugin crash reports even when direct interpreter commands succeed. Use the direct interpreter path `/home/hnxxzy/miniconda3/envs/ovvs/bin/python` rather than `conda run`.
