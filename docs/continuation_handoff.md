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
11. Executable reproduction pipeline added:
   - list steps: `PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python scripts/run_reproduction_pipeline.py --list`
   - runner: `scripts/run_reproduction_pipeline.py --run --skip-existing`
   - current pipeline covers official/Karpathy split extraction/eval, CKA proxy sweep, prompt sweep, VOC20/Context/ADE20K segmentation, K sweep, tau sweep, token usage ablation, anchor overlap, and baseline tracking.
12. COCO Karpathy test split completed:
   - tokens: `data/tokens/coco2014_karpathy_test_multicaption`
   - images/texts: `5000 / 25010`
   - output: `outputs/pal_k512_coco2014_full/coco_karpathy_test_multicaption_retrieval.json`
   - I2T/T2I R@1: `49.22 / 36.63`
13. Flickr30k Karpathy test split completed:
   - tokens: `data/tokens/flickr30k_karpathy_test_multicaption`
   - images/texts: `1000 / 5000`
   - output: `outputs/pal_k512_coco2014_full/flickr30k_karpathy_test_multicaption_retrieval.json`
   - I2T/T2I R@1: `67.40 / 50.96`
14. CKA proxy sweep completed:
   - output: `outputs/cka/coco_karpathy_layer_sweep.json`
   - best pair: `vision_layer=-1`, `text_layer=-2`, linear CKA `0.665336`
15. Prompt-template classification sweep completed:
   - output: `outputs/prompt_sweep/classification/summary.json`
   - best average top1: `46.09` vs paper `51.46`
   - best templates: STL10/Caltech101/EuroSAT close-up, CIFAR100 default photo, DTD cropped
16. VOC20 full-val segmentation completed:
   - output: `outputs/pal_k512_coco2014_full/voc20_segmentation_full.json`
   - samples: `1449`
   - foreground mIoU: `14.82` vs paper `32.30`
17. Context/ADE20K full segmentation completed:
   - Context output: `outputs/pal_k512_coco2014_full/context_segmentation_full.json`, mIoU `0.53` vs paper `25.50`
   - ADE20K output: `outputs/pal_k512_coco2014_full/ade20k_segmentation_full.json`, mIoU `1.47` vs paper `13.80`
18. K / `tau_p` / token-usage training sweeps completed:
   - K loss: `0.509355 -> 0.283415` from K32 to K512
   - tau loss: `0.252805 / 0.283415 / 0.335997 / 0.371047 / 0.407706` for `0.02 / 0.03 / 0.05 / 0.07 / 0.10`
   - token usage loss: global `0.740161`, mean `0.501132`, cap `0.283415`
19. Anchor overlap completed:
   - output: `outputs/analysis/coco_karpathy_anchor_overlap.json`
   - matched `0.517633` vs mismatched `0.436705`
20. Full pipeline result summary written:
   - `docs/pipeline_results_snapshot.md`
21. Tests currently pass:
   - command: `PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v`
   - result: `Ran 42 tests ... OK`

## Current active long-running process

No active Hermes-managed background process at this handoff point. `proc_fd7c67b922d5` completed normally.

## Next commands

Next useful commands are documentation/commit checks and optional downstream ablation evaluation. The existing K/tau/token checkpoints have only training-loss metrics; run retrieval/classification/segmentation evaluations per checkpoint before claiming paper ablation-table parity.

## Known gaps for paper-level parity

1. COCO/Flickr30k Karpathy retrieval split alignment is complete.
2. Prompt sweep is complete; mixed best templates improve classification average top1 to `46.09`, but do not close the full paper gap.
3. VOC20/Context/ADE20K full segmentation runs are complete but far below paper, especially Context/ADE20K; dense protocol debugging remains.
4. Ablation training runs are complete, but downstream ablation metrics remain to run.

## Important caveat

Conda may print plugin crash reports even when direct interpreter commands succeed. Use the direct interpreter path `/home/hnxxzy/miniconda3/envs/ovvs/bin/python` rather than `conda run`.
