# PAL Fine-Grained Relative Representation Reproduction Plan

> **For Hermes:** implement this plan task-by-task in the local conda Python environment.

**Goal:** Reproduce the core method from *Learning Relative Representations for Fine-Grained Multimodal Alignment with Limited Data* by implementing PAL (Projection-Free Anchor Learning) with token-to-anchor relative representations, CAP pooling, symmetric InfoNCE training, and retrieval/classification-ready evaluation hooks.

**Architecture:** Keep unimodal encoders frozen and train only modality-specific anchor matrices. Pre-extracted image/text token tensors feed a strict PAL module: cosine token-to-anchor similarities -> anchor-wise softmax over token positions -> pooled K-dimensional profiles -> L2 normalization -> symmetric contrastive loss. The reference `shiwonkim/bridge-anchors` repo is used as a scaffold/reference for anchors + contrastive training, but the implementation here is a clean PAL-strict path matching the paper.

**Tech Stack:** local conda interpreter `/home/hnxxzy/miniconda3/envs/ovvs/bin/python`, PyTorch, YAML/JSON configs, optional HuggingFace Transformers for full token extraction.

---

## Evidence gathered

- Paper PDF extracted to `docs/paper_text.txt`.
- Paper target settings:
  - Frozen encoders: DINOv2 ViT-L and RoBERTa-Large.
  - Training split: MS COCO 2014 train, about 80K first-caption image/text pairs.
  - PAL main hyperparameters: `K=512`, CAP temperature `0.03`, symmetric contrastive temperature `0.07`.
  - Evaluation contract: classification top-1 on STL10/CIFAR100/Caltech101/DTD/EuroSAT; retrieval R@1 on Flickr30k and COCO; segmentation foreground mIoU on VOC20/Context/ADE20K.
- Reference repo cloned to `external/bridge-anchors`; it has anchor/CAP code but also many experimental additions, so strict PAL will be implemented separately.
- Local conda probe:
  - `/home/hnxxzy/miniconda3/envs/ovvs/bin/python`: Python 3.11.15, torch 2.5.1+cu124, CUDA available, torchvision/PIL/YAML/numpy installed.
  - `conda activate`/`conda run` is currently unreliable in this TUI shell, so commands will call the conda interpreter path directly.
- Local data discovered:
  - COCO2014 raw: `/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/coco2014/raw`, 82,783 train images and `captions_train2014.json`.
  - Small local token cache for verification: `/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/artifacts/runs/pal_scoped_coco_retrieval_pilot_128x64/tokens`, tensors shaped image `(192,257,1024)`, text `(192,48,1024)`, mask `(192,48)`.

---

## Task 1: Create package skeleton and PAL tests first

**Objective:** Establish the strict PAL API and behavior through tests before implementation.

**Files:**
- Create: `src/pal_repro/__init__.py`
- Create: `src/pal_repro/models/__init__.py`
- Create: `src/pal_repro/models/pal.py`
- Create: `src/pal_repro/losses.py`
- Create: `tests/test_pal_core.py`

**Acceptance tests:**
- PAL forward accepts `(B,Tv,Dv)` image tokens, `(B,Tl,Dl)` text tokens, and text mask.
- Outputs are `(B,K)` and L2-normalized.
- Optional token similarities are shaped `(B,T,K)`.
- The only trainable parameters are `anchors_img` and `anchors_txt`.
- Symmetric InfoNCE returns a finite scalar and backpropagates to both anchor banks.
- All-masked text samples raise `ValueError`.

**Verification command:**

```bash
/home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
```

---

## Task 2: Implement strict PAL core

**Objective:** Match the paper equations (2)-(8).

**Files:**
- Modify: `src/pal_repro/models/pal.py`
- Modify: `src/pal_repro/losses.py`

**Implementation details:**
1. Initialize `anchors_img: (K,Dv)` and `anchors_txt: (K,Dl)` as `nn.Parameter`.
2. Normalize tokens and anchors for cosine similarities.
3. Compute `R_m = Z_m_normalized @ A_m_normalized.T`.
4. Apply CAP: `softmax(R_m / tau_p, dim=token)` independently for each anchor.
5. Pool each anchor dimension with `sum_t alpha[t,k] * R[t,k]`.
6. L2-normalize pooled profiles.
7. Train with symmetric InfoNCE over `image @ text.T / tau`.

---

## Task 3: Add tensor dataset, split, retrieval metrics, and training loop

**Objective:** Train/evaluate PAL on pre-extracted token tensors without re-running frozen encoders.

**Files:**
- Create: `src/pal_repro/data.py`
- Create: `src/pal_repro/eval.py`
- Create: `src/pal_repro/train.py`
- Create: `tests/test_training_smoke.py`

**Acceptance tests:**
- Tensor dataset loads image/text/mask `.pt` files and supports deterministic train/eval slices.
- Retrieval metrics compute R@1/R@5/R@10 for I2T and T2I.
- A tiny synthetic training epoch decreases or at least produces finite train/eval metrics and writes a checkpoint.

---

## Task 4: Add paper-grade config and optional token extraction script

**Objective:** Make the code runnable for both local smoke and full reproduction.

**Files:**
- Create: `configs/pal_strict.yaml`
- Create: `scripts/extract_coco_tokens.py`
- Create: `scripts/run_local_smoke.sh`
- Create: `README.md`

**Config values:**
- Main: `num_anchors: 512`, `pool_temperature: 0.03`, `contrastive_temperature: 0.07`.
- Encoders: `facebook/dinov2-large` and `roberta-large`.
- Data root: local COCO2014 raw path above.
- Smoke override: `num_anchors: 64`, `epochs: 2`, local 192-sample token cache.

---

## Task 5: Verify with local conda Python and local token cache

**Objective:** Produce real execution evidence.

**Commands:**

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m pal_repro.train --config configs/pal_strict.yaml --preset smoke --output-dir outputs/local_smoke --epochs 2 --batch-size 32 --num-anchors 64
```

**Done when:**
- Tests pass.
- Smoke training writes `outputs/local_smoke/metrics.json` and `outputs/local_smoke/checkpoint.pt`.
- Metrics include loss, I2T/T2I recall, train/eval split sizes, tensor paths, and parameter names.

---

## Paper-grade follow-up

Full paper-grade reproduction requires extracting all 82,783 COCO train token pairs with DINOv2 ViT-L/RoBERTa-Large, then running K=512 training and the complete evaluation matrix. This repository will make that route executable, but the current implementation milestone will be validated with the available 192-sample local token cache to avoid pretending a full multi-hour/multi-dataset run has been completed.
