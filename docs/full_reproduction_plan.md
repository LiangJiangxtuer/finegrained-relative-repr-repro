# Full PAL Paper Reproduction Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task when continuing beyond the current controller session.

**Goal:** Turn the bridge-anchors scaffold plus the strict PAL module into a full reproduction of all paper claims, datasets, tables, ablations, and anchor analyses.

**Architecture:** The reference `external/bridge-anchors` remains the base/scaffold for training style and retrieval utilities. The local `src/pal_repro` package is the PAL-strict layer that removes experimental extras and adds paper-contract experiment/evaluation code. Frozen encoder outputs are cached to disk, PAL trains only anchors, and downstream tasks consume cached tokens/features.

**Tech Stack:** Python 3.11 via `/home/hnxxzy/miniconda3/envs/ovvs/bin/python`, PyTorch/CUDA, HuggingFace Transformers, torchvision datasets, YAML experiment manifests.

---

## Task 1: Paper contract and target matrix

**Status:** Implemented.

**Files:**
- `docs/paper_claims_and_experiments.md`
- `configs/reproduction_matrix.yaml`
- `src/pal_repro/experiment_matrix.py`

**Verification:** `tests/test_eval_contract.py` checks that all paper tasks and ablations are represented.

## Task 2: PAL strict model and paper evaluation hooks

**Status:** Implemented for retrieval/classification primitives and mIoU primitive.

**Files:**
- `src/pal_repro/models/pal.py`
- `src/pal_repro/eval.py`
- `src/pal_repro/analysis.py`

**Implemented:**
- `encode_image` / `encode_text` modality-specific profiles.
- `evaluate_retrieval_model` for paired I2T/T2I R@K.
- `evaluate_zero_shot_classification` for class prompt tokens.
- `foreground_miou` for segmentation masks.
- `anchor_overlap_report` for Table 4-style top-k overlap and Dice.

## Task 3: Full COCO token extraction

**Status:** Completed for COCO2014 train first-caption pairs.

**Files:**
- `scripts/extract_coco_tokens.py`

**Command:**

```bash
bash scripts/run_full_coco_extraction.sh
```

This writes chunked fp16 tensors under `data/tokens/coco2014_full/chunks/`. Merge them before the current monolithic trainer:

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python scripts/merge_token_chunks.py data/tokens/coco2014_full
```

**Current output:** `data/tokens/coco2014_full` contains 82,783 merged fp16 image/text token pairs plus metadata.

## Task 4: K=512 full PAL training

**Status:** Completed for the main PAL run.

**Command:**

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m pal_repro.train \
  --config configs/pal_strict.yaml \
  --data-dir data/tokens/coco2014_full \
  --output-dir outputs/pal_k512_coco2014_full \
  --num-anchors 512 \
  --epochs 20 \
  --batch-size 128
```

**Engineering note:** `load_token_tensors` now preserves fp16 storage dtype and casts batches to fp32 on device to avoid doubling full-cache CPU memory.

**Current output:** `outputs/pal_k512_coco2014_full/checkpoint.pt` and `metrics.json`, with final train loss `0.28341474021328655` and trainable parameters limited to `anchors_img` / `anchors_txt`.

## Task 5: Downstream feature/token extraction and evaluation

**Status:** Completed for COCO 5K multi-caption, Flickr30k 1K multi-caption, and five zero-shot classification datasets. Segmentation datasets still need full paper-protocol runs.

**Implemented scripts / commands:**
- `scripts/extract_flickr30k_tokens.py` for zip-local Flickr30k retrieval tokens.
- `scripts/extract_coco_tokens.py --caption-policy all` for multi-caption COCO retrieval tokens.
- `scripts/evaluate_classification.py` for STL10/CIFAR100/Caltech101/DTD/EuroSAT.
- `python -m pal_repro.evaluate retrieval-multicaption ...` for multi-caption retrieval.

**Evaluation targets:** See `configs/reproduction_matrix.yaml`.

## Task 6: Segmentation pipeline

**Status:** Dense VOC20 code path exists and passes a 4-sample smoke run. Full VOC20/Context/ADE20K evaluation remains.

**Required behavior:**
1. Encode image patch tokens into patch PAL profiles or use token-sim matrices.
2. Encode class-name text prompts into class PAL profiles.
3. Compute patch-class similarity.
4. Reshape patch predictions to ViT grid, upsample to image/mask size.
5. Compute foreground mIoU for VOC20/Context/ADE20K.

## Task 7: Ablation runner

**Status:** Matrix exists; runner remains.

**Experiments:**
- token usage/CAP variants: global only, full tokens mean, full tokens CAP;
- K values: 32, 64, 128, 256, 512;
- `tau_p`: 0.02, 0.03, 0.05, 0.07, 0.10;
- data scaling: COCO80K vs COCO80K + COCO2017 30K.

## Task 8: Baselines

**Status:** Reference scaffold has Linear/MLP/FixedRelativeRep and other experimental modules; exact CSA/SAIL/FA parity is not yet implemented.

**Policy:** If the goal is to match PAL absolute numbers only, baselines are optional. If the goal is to reproduce every comparison table, implement/run CSA, LinearRS, MLPRS, SAIL, and FA or import official implementations where licenses permit.

## Task 9: Final paper-grade report

**Needed output:**
- `outputs/reports/table2_classification_retrieval.csv`
- `outputs/reports/table3_segmentation.csv`
- `outputs/reports/table5_token_cap_ablation.csv`
- `outputs/reports/table6_data_scaling.csv`
- `outputs/reports/table7_tau_ablation.csv`
- `outputs/reports/table4_anchor_overlap.csv`
- JSON with exact configs, seeds, local deviations, and gaps.

**Current interim report:** `REPRODUCTION_SUMMARY.md` records current progress, executable steps, measured results, gaps against the paper, and next reproduction steps.
