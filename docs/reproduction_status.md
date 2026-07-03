# Reproduction Status

## Completed

- Extracted the attached paper text to `docs/paper_text.txt`.
- Cloned reference repository `shiwonkim/bridge-anchors` into `external/bridge-anchors`.
- Wrote the implementation plan to `docs/reproduction_plan.md`.
- Implemented a strict PAL core in `src/pal_repro/models/pal.py`:
  - token-to-anchor cosine relative representations;
  - Cross-Attention Pooling with anchor-wise token softmax;
  - L2-normalized pooled profiles;
  - anchor-only trainable parameter boundary.
- Implemented symmetric InfoNCE in `src/pal_repro/losses.py`.
- Implemented token tensor loading/splitting, retrieval metrics, and a train/eval loop.
- Added `configs/pal_strict.yaml`, `scripts/run_local_smoke.sh`, and optional `scripts/extract_coco_tokens.py`.

## Verification evidence

### Unit tests

Command:

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
```

Result: `Ran 9 tests in 4.740s — OK`.

### Local smoke training

Command:

```bash
bash scripts/run_local_smoke.sh
```

Data: local 192-sample DINOv2-L/RoBERTa-L token cache at `/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/artifacts/runs/pal_scoped_coco_retrieval_pilot_128x64/tokens`.

Result files:

- `outputs/local_smoke/checkpoint.pt`
- `outputs/local_smoke/metrics.json`

Final smoke metrics on 64 held-out pairs:

- I2T R@1: `0.109375`
- I2T R@5: `0.3125`
- I2T R@10: `0.40625`
- T2I R@1: `0.046875`
- T2I R@5: `0.15625`
- T2I R@10: `0.25`
- Final train loss: `3.4403480887413025`
- Trainable parameters: `anchors_img`, `anchors_txt`

## Not yet paper-grade

The repository now has the executable PAL core and verification route, but the full paper-grade reproduction still requires:

1. full COCO2014 train token extraction for ~82,783 first-caption image/text pairs;
2. K=512 training over the full cache;
3. complete downstream evaluation on the paper's classification, retrieval, and segmentation benchmark matrix.
