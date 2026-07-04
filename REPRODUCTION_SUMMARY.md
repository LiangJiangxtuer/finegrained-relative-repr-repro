# PAL 论文复现进度报告

论文：**Learning Relative Representations for Fine-Grained Multimodal Alignment with Limited Data**  
方法：**Projection-Free Anchor Learning (PAL)**  
报告时间：2026-07-03 01:30:51 EDT  
仓库：`finegrained-relative-repr-repro`

> 当前结论：仓库已经完成严格 PAL 核心实现、COCO2014 约 80K 训练样本 token 缓存、K=512 主模型训练，以及主要 retrieval / zero-shot classification 评测。COCO/Flickr30k Karpathy official split 已对齐并复评；official retrieval 平均 R@1 达到论文目标的约 **86.17%**，分类平均 top-1 达到论文目标的约 **86.85%**。尚未完成 paper-grade segmentation 全量评测、K / `tau_p` / token-CAP 等消融实验、layer-specific CKA 复训和 baseline 对比。

---

## 1. 当前复现范围与进度

### 已完成

- 论文文本与主张梳理：
  - `docs/paper_text.txt`
  - `docs/paper_claims_and_experiments.md`
  - `docs/full_reproduction_plan.md`
  - `configs/reproduction_matrix.yaml`
- 参考基座：已克隆 `shiwonkim/bridge-anchors` 到 `external/bridge-anchors` 供对照。
- 严格 PAL 实现：
  - `src/pal_repro/models/pal.py`
  - `src/pal_repro/losses.py`
  - `src/pal_repro/train.py`
  - `src/pal_repro/eval.py`
  - `src/pal_repro/evaluate.py`
- 评测与分析支持：
  - retrieval R@K / multi-caption retrieval
  - zero-shot classification top-k
  - foreground mIoU primitive 与 VOC20 segmentation smoke runner
  - anchor overlap / Dice 分析接口
- 全量 COCO2014 train first-caption token cache：
  - `data/tokens/coco2014_full`
  - `image_tokens.pt`: `(82783, 257, 1024)` fp16，约 40.58 GiB
  - `text_tokens.pt`: `(82783, 64, 1024)` fp16，约 10.11 GiB
  - `text_mask.pt`: `(82783, 64)` bool
- K=512 主模型训练完成：
  - checkpoint: `outputs/pal_k512_coco2014_full/checkpoint.pt`
  - metrics: `outputs/pal_k512_coco2014_full/metrics.json`
  - trainable params: `anchors_img`, `anchors_txt`
- 已完成下游评测：
  - COCO val first-caption one-to-one retrieval proxy
  - COCO val 5K multi-caption retrieval
  - COCO Karpathy test 5K multi-caption retrieval
  - Flickr30k local 1K multi-caption retrieval
  - Flickr30k Karpathy test 1K multi-caption retrieval
  - STL10 / CIFAR100 / Caltech101 / DTD / EuroSAT zero-shot classification
  - VOC20 4-sample segmentation smoke
- 当前活跃后台任务：prompt-template classification sweep；后续 segmentation + ablation pipeline 已排队等待 prompt sweep 完成。

### 尚未完成

- VOC20 / Pascal Context / ADE20K segmentation 全量 paper protocol 评测。
- `K in [32, 64, 128, 256, 512]` anchor count 消融。
- `tau_p in [0.02, 0.03, 0.05, 0.07, 0.10]` CAP 温度消融。
- token usage 消融 pipeline 已支持 global / full-token mean / CAP，正式长跑任务已排队。
- COCO80K + COCO2017 30K data scaling 实验。
- 官方/Karpathy split 已完成 COCO/Flickr30k 对齐；prompt template sweep 已启动；CKA proxy sweep 已完成。
- CSA / LinearRS / MLPRS / SAIL / FA 等 baseline 行的完整对比。

---

## 2. 论文方法与实现对齐

论文核心主张：低数据量后验多模态对齐不应只对齐全局 pooled representation，而应保留图像 patch token 与文本 token 的细粒度结构；PAL 通过两个 modality-specific anchor bank 学习相对表示，并且不使用投影头。

当前实现对齐的公式路径：

1. 冻结 DINOv2 ViT-L 与 RoBERTa-Large。
2. 对 image/text token 和 modality anchors 做 L2 normalize。
3. 计算 token-to-anchor cosine similarity。
4. 用 Cross-Attention Pooling（CAP）在 token 维度做 anchor-wise softmax，默认 `tau_p=0.03`。
5. 得到 image/text 的 `(B, K)` relative profile，并再次 L2 normalize。
6. 用 symmetric InfoNCE 训练 image/text profile，默认 contrastive temperature `tau=0.07`。
7. 严格限制可训练参数为：`anchors_img` 与 `anchors_txt`。

默认主实验配置：

| 项目 | 当前复现配置 | 论文目标/描述 |
|---|---|---|
| Vision encoder | `facebook/dinov2-large` | DINOv2 ViT-L |
| Text encoder | `roberta-large` | RoBERTa-Large |
| Training data | COCO2014 train first-caption pairs | MS COCO 2014 train，约 80K pairs |
| Actual train pairs | 82,783 | 约 80K |
| Anchor count | 512 | 512 |
| CAP temperature | 0.03 | 0.03 |
| Trainable params | `anchors_img`, `anchors_txt` | projection-free anchors |

---

## 3. 可复现实验步骤

以下命令使用本地已验证 Python：

```bash
export PY=/home/hnxxzy/miniconda3/envs/ovvs/bin/python
export PYTHONPATH=src
```

> 注意：当前环境中 `conda activate` / `conda run` 可能触发 conda activate TypeError；推荐直接调用上述解释器路径。

### 3.1 运行测试

```bash
PYTHONPATH=src $PY -m unittest discover -s tests -v
```

最近记录结果：`Ran 26 tests ... OK`。

### 3.2 提取 COCO2014 train token cache

```bash
PYTHONUNBUFFERED=1 BATCH_SIZE=8 CHUNK_SIZE=2048 bash scripts/run_full_coco_extraction.sh
```

该脚本使用：

- captions: `/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/coco2014/raw/annotations/captions_train2014.json`
- images: `/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/coco2014/raw/train2014`
- output: `data/tokens/coco2014_full`
- encoder: DINOv2-L / RoBERTa-Large
- storage: chunked fp16

当前 trainer 读取 monolithic tensors，因此 chunk 提取后需合并：

```bash
PYTHONPATH=src $PY scripts/merge_token_chunks.py data/tokens/coco2014_full
```

### 3.3 训练 K=512 PAL 主模型

```bash
PYTHONUNBUFFERED=1 EPOCHS=20 BATCH_SIZE=128 TRAIN_SIZE=82783 bash scripts/run_full_pal_train.sh
```

输出：

- `outputs/pal_k512_coco2014_full/checkpoint.pt`
- `outputs/pal_k512_coco2014_full/metrics.json`

本次训练使用全部 82,783 个 COCO train pairs，因此没有内部 held-out eval split（`eval_size: 0`）。

### 3.4 Retrieval 评测

COCO 5K multi-caption 示例：

```bash
PYTHONPATH=src $PY -m pal_repro.evaluate retrieval-multicaption \
  --checkpoint outputs/pal_k512_coco2014_full/checkpoint.pt \
  --token-dir data/tokens/coco2014_val_5k_multicaption \
  --output outputs/pal_k512_coco2014_full/coco_val_5k_multicaption_retrieval.json \
  --batch-size 256
```

Flickr30k 1K multi-caption token 提取与评测：

```bash
PYTHONPATH=src $PY scripts/extract_flickr30k_tokens.py \
  --zip-path /home/hnxxzy/Downloads/Flickr30k.zip \
  --output-dir data/tokens/flickr30k_1k_multicaption \
  --caption-policy all \
  --limit 1000 \
  --batch-size 8 \
  --chunk-size 2048

PYTHONPATH=src $PY -m pal_repro.evaluate retrieval-multicaption \
  --checkpoint outputs/pal_k512_coco2014_full/checkpoint.pt \
  --token-dir data/tokens/flickr30k_1k_multicaption \
  --output outputs/pal_k512_coco2014_full/flickr30k_1k_multicaption_retrieval.json \
  --batch-size 256
```

### 3.5 Zero-shot classification 评测

单数据集示例：

```bash
PYTHONPATH=src $PY scripts/evaluate_classification.py \
  --dataset STL10 \
  --checkpoint outputs/pal_k512_coco2014_full/checkpoint.pt \
  --output outputs/pal_k512_coco2014_full/stl10_classification.json \
  --batch-size 64
```

已完成数据集：`STL10`, `CIFAR100`, `Caltech101`, `DTD`, `EuroSAT`。

### 3.6 Segmentation smoke

当前只完成了 VOC20 小样本 smoke，不是论文全量 segmentation 结果：

```bash
PYTHONPATH=src $PY scripts/evaluate_segmentation.py \
  --dataset VOC20 \
  --checkpoint outputs/pal_k512_coco2014_full/checkpoint.pt \
  --output outputs/pal_k512_coco2014_full/voc20_segmentation_smoke.json \
  --batch-size 2 \
  --limit 4
```

---

## 4. 训练结果

| 指标 | 当前结果 |
|---|---:|
| Train samples | 82,783 |
| Epochs | 20 |
| Batch size | 128 |
| Initial train loss | 0.856644159491 |
| Final train loss | 0.283414740213 |
| Eval split | 0 |
| Trainable parameter names | `anchors_img`, `anchors_txt` |

解释：训练 loss 稳定下降，说明 PAL anchor-only training 路径可正常优化；但训练 loss 不是论文主指标，paper-grade 结论必须以后续 downstream evaluation 为准。

---

## 5. 与论文指标对比

### 5.1 Retrieval R@1

| Protocol | Images | Texts | Local I2T R@1 | Paper I2T R@1 | Gap | Relative | Local T2I R@1 | Paper T2I R@1 | Gap | Relative | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| COCO val 5K multi-caption | 5,000 | 25,021 | 49.82 | 56.30 | -6.48 | 88.49% | 37.06 | 42.60 | -5.54 | 87.01% | 接近论文 protocol，但当前为 seed=42 local 5K subset，官方 split 未确认 |
| COCO Karpathy test 5K multi-caption | 5,000 | 25,010 | 49.22 | 56.30 | -7.08 | 87.42% | 36.63 | 42.60 | -5.97 | 85.99% | 标准 Karpathy test split，当前推荐作为 COCO paper-protocol 对比行 |
| Flickr30k local 1K multi-caption | 1,000 | 5,000 | 67.80 | 76.30 | -8.50 | 88.86% | 52.80 | 61.80 | -9.00 | 85.44% | 使用 `/home/hnxxzy/Downloads/Flickr30k.zip`，seed=42 local 1K subset，官方 split 未确认 |
| Flickr30k Karpathy test 1K multi-caption | 1,000 | 5,000 | 67.40 | 76.30 | -8.90 | 88.34% | 50.96 | 61.80 | -10.84 | 82.46% | 标准 Karpathy test split，当前推荐作为 Flickr30k paper-protocol 对比行 |

Retrieval R@1 四项平均（preferred Karpathy rows）：

| Local avg R@1 | Paper avg R@1 | Gap | Relative |
|---:|---:|---:|---:|
| 51.05 | 59.25 | -8.20 | 86.17% |

额外 proxy 结果：COCO val first-caption one-to-one 使用 40,504-way 单 caption 严格检索，I2T/T2I R@1 为 `15.22 / 15.01`。这不是论文 multi-caption protocol，只作为排查与压力测试参考。

### 5.2 Zero-shot classification top-1

| Dataset | N | Local top-1 | Local top-5 | Paper top-1 | Gap | Relative |
|---|---:|---:|---:|---:|---:|---:|
| STL10 | 8,000 | 91.96 | 99.96 | 95.30 | -3.34 | 96.50% |
| CIFAR100 | 10,000 | 42.58 | 71.47 | 48.80 | -6.22 | 87.25% |
| Caltech101 | 8,677 | 45.63 | 67.82 | 60.90 | -15.27 | 74.92% |
| DTD | 1,880 | 15.21 | 33.94 | 17.70 | -2.49 | 85.95% |
| EuroSAT | 27,000 | 28.08 | 72.46 | 34.60 | -6.52 | 81.16% |
| **Average** | - | **44.69** | - | **51.46** | **-6.77** | **86.85%** |

分类结果分析：

- STL10 最接近论文，达到论文 top-1 的 96.50%。
- Caltech101 差距最大（-15.27），优先排查 split/protocol、class-name mapping、prompt template 与图像预处理。
- 当前 prompt template 是 `a photo of {class_name}`；论文 PDF 文本未明确 prompt template，因此 prompt engineering 可能解释一部分差距。
- 当前使用 final hidden tokens；论文提到 CKA-based layer selection，但 PDF 文本中未恢复精确 layer indices，这可能影响分类与检索整体差距。

### 5.3 Segmentation foreground mIoU

| Dataset | Current run | Samples | Local mIoU | Paper target | Gap | Relative | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| VOC20 | smoke | 4 | 1.69 | 32.30 | -30.61 | 5.23% | 仅验证 runner 可执行，不能代表论文结果 |
| Context | queued | - | - | 25.50 | - | - | loader 已实现，full-val job 已在 downstream pipeline 中排队 |
| ADE20K | queued | - | - | 13.80 | - | - | loader 已实现，full-val job 已在 downstream pipeline 中排队 |

Segmentation 结果分析：当前 VOC20 仅跑了 4 张图的 smoke，主要用于验证 dense patch profile -> class prompt similarity -> mask upsample -> foreground mIoU 的代码路径，没有统计意义。下一步需要补齐全量 VOC20 / Context / ADE20K protocol，并检查是否需要背景类、ignore-index 处理、prompt template、mask resize 和 paper foreground-mIoU 口径对齐。

### 5.4 消融实验状态

| 实验组 | 论文目标 | 当前状态 | 下一步 |
|---|---|---|---|
| Token usage + CAP | global only / full tokens / CAP 三行平均指标 | 未运行 | 在同一 cache 上实现 pooling variant runner，并复现 avg cls / avg ret / avg seg |
| Anchor count K | `32, 64, 128, 256, 512` | 仅 K=512 完成 | 复用 COCO token cache 训练 K sweep，并跑核心下游指标 |
| Data scaling | COCO80K vs COCO80K + COCO2017 30K | 未运行 | 准备 COCO2017 30K 追加数据与 token cache |
| CAP temperature `tau_p` | `0.02, 0.03, 0.05, 0.07, 0.10` | 仅 `0.03` 完成 | 固定 K=512 后训练温度 sweep |
| Anchor overlap | Flickr30k / COCO matched vs mismatched overlap & Dice | 接口存在，未产出表 | 用已训练 checkpoint 计算 top-5 activated anchors overlap |
| Qualitative attention | anchor heatmap / caption attention > 0.5 | 未运行 | 输出可视化样例图和 markdown/csv 索引 |

---

## 6. 当前差距的主要可能原因

1. **评测 split 未完全对齐**  
   COCO/Flickr30k Karpathy test split 已对齐并复评；旧 seed=42 local subset 仅作为历史 proxy 保留。

2. **CKA layer selection 尚未复现**
   已完成 128-pair CKA proxy sweep，当前最佳 pair 为 `vision_layer=-1`, `text_layer=-2`，但还未做 layer-specific full token extraction + retraining。  
   当前实现默认使用 DINOv2-L / RoBERTa-Large final hidden states。论文提到基于 CKA 选择 encoder layer，但未在已抽取文本中恢复精确 layer indices。若论文实际使用中间层，当前差距可能来自 layer mismatch。

3. **Prompt template 未对齐**  
   分类与 segmentation 目前使用 `a photo of {class_name}`。论文如果使用 dataset-specific prompt ensemble 或不同模板，可能显著影响 zero-shot classification，尤其是 Caltech101 / EuroSAT。

4. **Segmentation pipeline 仍处于 smoke 阶段**  
   Dense segmentation 还没有全量运行，也没有完成 Context / ADE20K；当前 mIoU 不能用于论文 parity 判断。

5. **Baseline 和 ablation 尚未完成**  
   当前只比较 PAL 主模型 target row；若目标是完整复现论文所有表格，还需要实现或导入 CSA / LinearRS / MLPRS / SAIL / FA，并运行全部 ablation。

---

## 7. 后续复现步骤建议

按优先级建议如下：

1. **锁定官方评测 protocol**
   - 确认 COCO / Flickr30k 是否使用 Karpathy split 或论文自定义 split。
   - 将当前 seed=42 local subset 替换为官方 split；保留 local subset 结果作为 proxy。

2. **复现 CKA layer selection**
   - 从论文、附录或作者实现线索恢复 DINOv2/RoBERTa layer indices。
   - 如无法恢复，至少做 layer sweep 并在报告中标注 best/reported deviation。

3. **分类误差排查**
   - 对 `a photo of {class_name}`、dataset-specific prompt、prompt ensemble 做小规模 sweep。
   - 优先排查 Caltech101 的 split 与 label mapping。
   - 输出每个数据集 per-class accuracy，定位主要失败类别。

4. **完成 segmentation 主表**
   - 先将 VOC20 从 4-sample smoke 扩展到 full val。
   - 实现/验证 Pascal Context 与 ADE20K loader。
   - 对齐 foreground mIoU 口径、背景类处理、ignore label、mask resize。

5. **运行主消融实验**
   - K sweep：`32, 64, 128, 256, 512`。
   - `tau_p` sweep：`0.02, 0.03, 0.05, 0.07, 0.10`。
   - Token usage：global only / full-token mean / full-token CAP。

6. **补齐 anchor analysis**
   - 用已完成的 COCO/Flickr retrieval pairs 计算 top-5 anchor overlap 与 Dice。
   - 生成定性 attention heatmaps 与 caption word attention 可视化。

7. **决定是否复现 baseline rows**
   - 如果目标是 PAL absolute parity，可先完成 PAL 主表和 ablation。
   - 如果目标是完整论文表格，需要额外实现/运行 CSA、LinearRS、MLPRS、SAIL、FA。

8. **整理最终 GitHub 提交材料**
   - 保留源码、配置、测试、文档。
   - 不提交大文件：`outputs/`, `data/tokens/`, `*.pt`, `*.pth` 已在 `.gitignore` 中排除。
   - `external/bridge-anchors` 建议以 README 链接或 git submodule 说明方式引用，不建议直接 vendor 全量第三方仓库。

---

## 8. 建议提交到 GitHub 的文档与文件结构

建议随代码提交：

```text
README.md
REPRODUCTION_SUMMARY.md
configs/
  data_manifest.local.yaml
  pal_strict.yaml
  reproduction_matrix.yaml
docs/
  paper_claims_and_experiments.md
  full_reproduction_plan.md
  full_reproduction_status.md
  continuation_handoff.md
  results_snapshot.md
scripts/
  extract_coco_tokens.py
  extract_flickr30k_tokens.py
  merge_token_chunks.py
  run_full_coco_extraction.sh
  run_full_pal_train.sh
  run_local_smoke.sh
  evaluate_classification.py
  evaluate_segmentation.py
src/pal_repro/
tests/
```

不建议提交：

```text
outputs/
data/tokens/
*.pt
*.pth
external/bridge-anchors/.git/
```

---

## 9. 证据文件索引

- 训练指标：`outputs/pal_k512_coco2014_full/metrics.json`
- COCO 5K multi-caption retrieval：`outputs/pal_k512_coco2014_full/coco_val_5k_multicaption_retrieval.json`
- COCO first-caption proxy retrieval：`outputs/pal_k512_coco2014_full/coco_val_first_caption_retrieval.json`
- Flickr30k 1K multi-caption retrieval：`outputs/pal_k512_coco2014_full/flickr30k_1k_multicaption_retrieval.json`
- 分类结果：
  - `outputs/pal_k512_coco2014_full/stl10_classification.json`
  - `outputs/pal_k512_coco2014_full/cifar100_classification.json`
  - `outputs/pal_k512_coco2014_full/caltech101_classification.json`
  - `outputs/pal_k512_coco2014_full/dtd_classification.json`
  - `outputs/pal_k512_coco2014_full/eurosat_classification.json`
- Segmentation smoke：`outputs/pal_k512_coco2014_full/voc20_segmentation_smoke.json`
- 已整理结果快照：`docs/results_snapshot.md`
- 续接上下文：`docs/continuation_handoff.md`
