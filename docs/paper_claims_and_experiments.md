# Paper Claim / Formula / Experiment Analysis

Paper: **Learning Relative Representations for Fine-Grained Multimodal Alignment with Limited Data**.

## Main thesis

The paper argues that low-data post-hoc multimodal alignment should align **fine-grained token structure**, not only global pooled representations. It claims that a projection-free anchor method can outperform projector-based alignment while training only modality-specific anchor matrices.

## Claimed innovation

1. **Projection-Free Anchor Learning (PAL)**
   - Does not learn image-to-text or text-to-image projection heads.
   - Learns only two anchor banks, one in the frozen visual feature space and one in the frozen language feature space.
   - Produces comparable cross-modal representations through token-to-anchor similarity profiles.

2. **Token-level relative representations**
   - Image patch tokens and text tokens are represented by cosine similarities to modality-specific anchors.
   - This is a learnable extension of relative representations / fixed anchors.

3. **Cross-Attention Pooling (CAP)**
   - No query/key/value projections.
   - For each anchor, applies softmax over token positions using token-to-anchor similarity divided by `tau_p`.
   - The anchor-wise weighted sum produces a single K-dimensional relative profile per sample.

4. **Parameter efficiency**
   - With DINOv2 ViT-L and RoBERTa-Large dimensions `D_v=D_l=1024`, K=512 gives `512*1024*2 = 1,048,576` trainable anchor parameters.
   - Paper reports this as linear-layer-scale while still token-level.

## Core formulas to reproduce

Let frozen encoders produce token matrices:

- Visual: `Z_v in R^(T_v x D_v)`
- Text: `Z_l in R^(T_l x D_l)`

For modality `m in {v,l}`, learn anchors:

```text
A_m = [a_m^1, ..., a_m^K]^T in R^(K x D_m)
```

Token-to-anchor cosine similarity:

```text
sim(z_m^t, a_m^k) = (z_m^t)^T a_m^k / (||z_m^t||_2 ||a_m^k||_2)
```

Token-level relative representation:

```text
r_m^t = [sim(z_m^t, a_m^1), ..., sim(z_m^t, a_m^K)] in R^K
```

Stacking all tokens:

```text
R_m in R^(T_m x K)
```

CAP attention for token `t` and anchor `k`:

```text
alpha_m,t,k = exp(R_m[t,k] / tau_p) / sum_t' exp(R_m[t',k] / tau_p)
```

Pooled profile:

```text
p_m[k] = sum_t alpha_m,t,k * R_m[t,k]
h_m = p_m / ||p_m||_2
```

Symmetric contrastive objective over batch size `B`:

```text
L_con = (1 / 2B) * sum_i [
  -log exp(h_v,i^T h_l,i / tau) / sum_j exp(h_v,i^T h_l,j / tau)
  -log exp(h_l,i^T h_v,i / tau) / sum_j exp(h_l,i^T h_v,j / tau)
]
```

Default hyperparameters:

- `K=512`
- `tau_p=0.03`
- contrastive `tau=0.07` (standard in reference scaffold; paper formula leaves it generic)

## Main experiment matrix

### Training data

- MS COCO 2014 train split.
- Paper says approximately 80K image-text pairs.
- Local first-caption construction yields 82,783 image-text pairs.

### Encoders

- Vision: DINOv2 ViT-L, frozen (`facebook/dinov2-large` in HF naming)
- Language: RoBERTa-Large, frozen (`roberta-large`)
- Paper mentions layer selection by CKA following prior work, but does not disclose exact layer indices in the PDF text. The current reproduction uses final hidden tokens by default and treats exact CKA layer selection as a tracked reproduction-risk item.

### Main metrics to match

Classification top-1 (%):

| Dataset | PAL target |
| --- | ---: |
| STL10 | 95.3 |
| CIFAR100 | 48.8 |
| Caltech101 | 60.9 |
| DTD | 17.7 |
| EuroSAT | 34.6 |

Retrieval R@1 (%):

| Dataset | I2T | T2I |
| --- | ---: | ---: |
| Flickr30k | 76.3 | 61.8 |
| COCO | 56.3 | 42.6 |

Segmentation foreground mIoU (%):

| Dataset | PAL target |
| --- | ---: |
| VOC20 | 32.3 |
| Context | 25.5 |
| ADE20K | 13.8 |

### Ablations to reproduce

1. **Token usage + CAP**

| Variant | Avg cls | Avg ret | Avg seg |
| --- | ---: | ---: | ---: |
| Global only | 48.4 | 43.9 | 7.3 |
| + Full tokens | 49.3 | 48.4 | 16.3 |
| + CAP | 51.5 | 59.3 | 23.9 |

2. **Number of anchors K**

`K in [32, 64, 128, 256, 512]`. PDF text provides qualitative Figure 4 trends and not exact all-point values.

3. **Scaling data**

| Source | Avg cls | Avg ret | Avg seg |
| --- | ---: | ---: | ---: |
| COCO 80K | 51.5 | 59.3 | 23.9 |
| + COCO 30K | 53.0 | 61.9 | 25.5 |

4. **CAP temperature**

| tau_p | Avg cls | Avg ret | Avg seg |
| ---: | ---: | ---: | ---: |
| 0.02 | 51.1 | 58.8 | 21.6 |
| 0.03 | 51.5 | 59.3 | 23.9 |
| 0.05 | 50.4 | 57.9 | 23.2 |
| 0.07 | 50.2 | 55.5 | 21.0 |
| 0.10 | 49.7 | 52.9 | 18.5 |

5. **Anchor overlap analysis**

Top-5 activated anchors, matched vs mismatched:

| Dataset | Hard overlap matched/mismatched | Dice matched/mismatched |
| --- | --- | --- |
| Flickr30k | 0.226 / 0.059 | 0.212 / 0.050 |
| COCO | 0.218 / 0.028 | 0.211 / 0.024 |

6. **Qualitative anchor attention**

Visualize anchor attention heatmaps and caption words with attention value greater than 0.5.

## Engineering gap list

Already implemented in this repo:

- Strict PAL module with anchor-only trainable parameters.
- Symmetric InfoNCE.
- Token tensor loading and smoke training.
- Retrieval/classification model-level evaluation utilities.
- Foreground mIoU primitive.
- Experiment matrix YAML with paper targets.

Still required for full paper-grade reproduction:

1. Full COCO2014 token extraction in chunked fp16 format.
2. Full K=512 training from all COCO2014 train pairs.
3. Downstream token extraction for STL10/CIFAR100/Caltech101/DTD/EuroSAT, COCO retrieval, Flickr30k retrieval, VOC20/Context/ADE20K segmentation.
4. Segmentation prediction pipeline from PAL patch profiles to dense class masks.
5. CKA layer-selection reproduction or explicit final-layer deviation report.
6. Baseline implementations/runs for CSA, LinearRS, MLPRS, SAIL, FA if exact comparative tables are required rather than PAL-only target matching.
