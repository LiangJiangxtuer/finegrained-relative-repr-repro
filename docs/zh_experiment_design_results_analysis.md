# PAL 论文复现实验设计、结果与分析（中文整理）

更新时间：2026-07-06

本文档按阶段整理 `/home/hnxxzy/finegrained-relative-repr-repro` 的论文复现工作，记录每一阶段的实验设计、实际输出、最佳/最新结果、分析结论与后续工作。所有指标均来自本地真实输出文件；大文件产物位于 `outputs/`、`data/tokens/` 等 ignored 路径，不随 Git 提交。

## 0. 项目目标与当前结论

目标论文：**Learning Relative Representations for Fine-Grained Multimodal Alignment**。本仓库实现论文中的 **Projection-Free Anchor Learning (PAL)** 路径：冻结 DINOv2-L / RoBERTa-Large encoder，仅训练 image/text anchors，通过 token-to-anchor 相似度与 Cross-Attention Pooling (CAP) 得到相对表示，再用 symmetric InfoNCE 对齐。

当前总体状态：

- 已完成：PAL 核心实现、COCO2014 全量 token extraction、K=512 主模型训练、COCO/Flickr30k Karpathy retrieval、5 个分类数据集 zero-shot、VOC20/Context/ADE20K corrected segmentation rerun、K/`tau_p`/token usage 训练型 sweep、K/`tau_p`/token usage downstream retrieval ablation、full classification ablation、corrected 64-sample segmentation probes、selected full corrected segmentation reruns、CKA proxy、anchor-overlap 分析。
- 最接近论文的部分：retrieval 与 classification；平均约达到论文指标的 86%~90%。
- 最大差距：dense segmentation，尤其 ADE20K；selected full `tau_p=0.07` + ADE20K clean aliases / `--ignore-zero` / recovered `last_hidden_state` dense tokens 的平均 mIoU 已达 `23.38`（论文平均 `23.87` 的 `97.94%`），但未校准 ADE20K 仍为 `10.55` vs 论文 `13.80`。诊断 targeted group calibration 将 ADE20K 提升到 `11.47`、平均 `23.68`（`99.23%`），但该行是 validation-informed。
- 当前主要待办：按需补齐剩余 checkpoint 的 VOC20/Context full segmentation rows；如需把 calibration 当主结果则增加 held-out calibration protocol；Context 进一步 dense-token layer / prompt ensemble，严格 CKA layer selection 与 baseline rows。

## 1. 实验环境与数据准备

### 1.1 环境设计

- Python：`/home/hnxxzy/miniconda3/envs/ovvs/bin/python`
- 运行方式：统一设置 `PYTHONPATH=src`。
- PyTorch：`2.5.1+cu124`
- GPU：NVIDIA GeForce RTX 4090 24GB
- 注意：当前 TUI 下 `conda activate`/`conda run` 可能触发 conda activate TypeError，因此所有命令直接调用解释器路径。

### 1.2 数据与 token cache

| 数据/产物 | 路径 | 状态 |
|---|---|---|
| COCO2014 raw | `/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/coco2014/raw` | 本地可用 |
| Flickr30k zip | `/home/hnxxzy/Downloads/Flickr30k.zip` | 本地可用，脚本直接读 zip |
| COCO2014 full train token cache | `data/tokens/coco2014_full` | 已完成 |
| COCO Karpathy test tokens | `data/tokens/coco2014_karpathy_test_multicaption` | 已完成 |
| Flickr30k Karpathy test tokens | `data/tokens/flickr30k_karpathy_test_multicaption` | 已完成 |
| Segmentation datasets | `configs/data_manifest.local.yaml` 中记录 | VOC20/Context/ADE20K 已跑通 |

COCO2014 full train token cache 规模：

| Tensor | Shape | dtype | 说明 |
|---|---:|---|---|
| `image_tokens.pt` | `(82783, 257, 1024)` | fp16 | DINOv2-L CLS + patch tokens |
| `text_tokens.pt` | `(82783, 64, 1024)` | fp16 | RoBERTa-L tokens |
| `text_mask.pt` | `(82783, 64)` | bool | 文本 attention mask |

## 2. PAL 方法实现阶段

### 2.1 实验设计

严格实现论文主路径，避免将参考仓库中的 projector/router/fixed anchor/auxiliary loss 混入 PAL：

1. 冻结 DINOv2-L 与 RoBERTa-Large encoder。
2. 对 image/text token 与 modality anchors 做 L2 normalize。
3. 计算 token-to-anchor cosine similarity。
4. 使用 CAP：对每个 anchor 在 token 维度做 softmax，温度 `tau_p=0.03`。
5. 每个 anchor 得到 weighted token similarity profile，形成 `(B, K)` 相对表示。
6. 对 image/text profile 归一化。
7. 用 symmetric InfoNCE 训练，contrastive temperature `tau=0.07`。

### 2.2 代码与测试

核心文件：

- `src/pal_repro/models/pal.py`
- `src/pal_repro/losses.py`
- `src/pal_repro/train.py`
- `src/pal_repro/evaluate.py`
- `src/pal_repro/eval.py`

约束：checkpoint 中只有 `anchors_img` 与 `anchors_txt` 是可训练参数。

当前完整测试：

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
```

最近结果：`Ran 63 tests in 0.206s OK`。

## 3. 主模型训练阶段

### 3.1 实验设计

| 项 | 设置 |
|---|---:|
| 训练数据 | COCO2014 train first-caption pairs |
| 样本数 | 82,783 |
| Image/Text dim | 1024 / 1024 |
| Anchor count | K=512 |
| CAP temperature | `tau_p=0.03` |
| Contrastive temperature | `tau=0.07` |
| Epochs | 20 |
| Batch size | 128 |
| 训练参数 | `anchors_img`, `anchors_txt` |

### 3.2 结果

| 指标 | 结果 |
|---|---:|
| Final train loss | `0.28341474021328655` |
| Train samples | `82,783` |
| Eval split | `0`，使用全部训练样本 |
| Checkpoint | `outputs/pal_k512_coco2014_full/checkpoint.pt` |
| Metrics | `outputs/pal_k512_coco2014_full/metrics.json` |

分析：训练 loss 稳定下降，anchor-only 路径可正常优化；但训练 loss 不是论文主指标，必须以后续 retrieval / classification / segmentation 为准。

## 4. Retrieval 阶段

### 4.1 实验设计

已实现 multi-caption retrieval 评测，优先使用 Karpathy test split 作为 paper-protocol comparison。历史 local subset/proxy 仍保留，但不作为最终 paper-parity 主表。

### 4.2 最佳/最新结果

| Protocol | Images | Texts | I2T R@1 | T2I R@1 | Paper I2T/T2I R@1 | 相对水平 | 输出 |
|---|---:|---:|---:|---:|---|---:|---|
| COCO Karpathy test | 5,000 | 25,010 | 49.22 | 36.63 | 56.30 / 42.60 | 87.42% / 85.99% | `outputs/pal_k512_coco2014_full/coco_karpathy_test_multicaption_retrieval.json` |
| Flickr30k Karpathy test | 1,000 | 5,000 | 67.40 | 50.96 | 76.30 / 61.80 | 88.34% / 82.46% | `outputs/pal_k512_coco2014_full/flickr30k_karpathy_test_multicaption_retrieval.json` |
| COCO val 5K proxy | 5,000 | 25,021 | 49.82 | 37.06 | 56.30 / 42.60 | 88.49% / 87.01% | `outputs/pal_k512_coco2014_full/coco_val_5k_multicaption_retrieval.json` |
| Flickr30k local 1K proxy | 1,000 | 5,000 | 67.80 | 52.80 | 76.30 / 61.80 | 88.86% / 85.44% | `outputs/pal_k512_coco2014_full/flickr30k_1k_multicaption_retrieval.json` |

Karpathy 四项 R@1 平均：

| Local avg R@1 | Paper avg R@1 | Gap | Relative |
|---:|---:|---:|---:|
| 51.05 | 59.25 | -8.20 | 86.17% |

分析：retrieval 是当前最接近论文的主任务之一。仍有 8%~18% 相对差距，可能来自 CKA layer selection、encoder layer choice、训练细节或官方实现未公开的协议差异。

## 5. Zero-shot classification 阶段

### 5.1 实验设计

数据集：STL10、CIFAR100、Caltech101、DTD、EuroSAT。先跑默认 prompt，再做 4 个模板 sweep：

- `a photo of {class_name}`
- `a cropped photo of {class_name}`
- `a close-up photo of {class_name}`
- `a clean photo of {class_name}`

### 5.2 最佳 prompt sweep 结果

| Dataset | Best template | Best top-1 | Paper top-1 | Gap | Relative |
|---|---|---:|---:|---:|---:|
| STL10 | `a close-up photo of {class_name}` | 92.15 | 95.30 | -3.15 | 96.69% |
| CIFAR100 | `a photo of {class_name}` | 42.58 | 48.80 | -6.22 | 87.25% |
| Caltech101 | `a close-up photo of {class_name}` | 48.84 | 60.90 | -12.06 | 80.20% |
| DTD | `a cropped photo of {class_name}` | 16.54 | 17.70 | -1.16 | 93.46% |
| EuroSAT | `a close-up photo of {class_name}` | 30.32 | 34.60 | -4.28 | 87.64% |
| Average | mixed best templates | 46.09 | 51.46 | -5.37 | 89.56% |

输出：`outputs/prompt_sweep/classification/summary.json`。

分析：prompt sweep 将平均 top-1 从 `44.69` 提升到 `46.09`，说明 prompt 选择能解释部分差距，但无法完全闭合论文指标。Caltech101 仍是分类任务中的主要短板，可能需要 split/class mapping/prompt ensemble 进一步排查。

## 6. Segmentation 阶段

### 6.1 初始实验设计与问题

初始 full segmentation 跑通了 VOC20、Pascal Context、ADE20K，但发现指标异常低。初始实现将 patch logits 上采样到原始 mask 尺寸；而本地 DINOv2 image processor 实际对图像做：

```text
do_resize=True, shortest_edge=256
do_center_crop=True, crop_size=224x224
```

因此 image tokens 对应的是 processor frame，而不是原图 frame。同时 Pascal Context 初始使用 all-459 protocol，与常见 common-59 protocol 不一致。

### 6.2 corrected protocol 设计

已加入正式 evaluator 参数：

- `--target-frame {original,processor}`：显式控制 target mask 几何 frame。
- `--context-protocol {all459,common59}`：显式控制 Pascal Context 459 类或 common-59 protocol。

新增/修改文件：

- `scripts/evaluate_segmentation.py`
- `scripts/diagnose_segmentation_protocol.py`
- `src/pal_repro/segmentation.py`
- `tests/test_segmentation_support.py`
- `docs/segmentation_debug_notes.md`

### 6.3 corrected full rerun 结果

| Dataset | Corrected protocol | Samples | Classes | Historical mIoU | Corrected mIoU | Paper | Delta vs old | Relative to paper | 输出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| VOC20 | processor frame | 1,449 | 20 | 14.82 | 20.58 | 32.30 | +5.75 | 63.71% | `outputs/pal_k512_coco2014_full/voc20_segmentation_processor_full.json` |
| Pascal Context | processor frame + common59 | 10,103 | 59 | 0.53 | 11.23 | 25.50 | +10.70 | 44.05% | `outputs/pal_k512_coco2014_full/context_segmentation_common59_processor_full.json` |
| ADE20K | processor frame | 2,000 | 150 | 1.47 | 2.19 | 13.80 | +0.72 | 15.87% | `outputs/pal_k512_coco2014_full/ade20k_segmentation_processor_full.json` |
| Average | - | - | - | 5.61 | 11.33 | 23.87 | +5.72 | 47.48% | - |

日志：`outputs/logs/corrected_segmentation_rerun_20260705_104725.log`。

### 6.4 分析结论

- Corrected protocol 使 segmentation 平均 mIoU 从 `5.61` 提升到 `11.33`，几乎翻倍。
- Context 从 `0.53` 提升到 `11.23`，说明 all-459 + original-mask frame 是主要 protocol mismatch。
- VOC20 从 `14.82` 提升到 `20.58`，说明 processor-frame target 对 dense alignment 很重要。
- ADE20K 仅从 `1.47` 提升到 `2.19`，仍显著低于论文；下一步不应继续盲目 full rerun，而应做 16-64 sample prompt/name/layer probes。

## 7. Ablation / sweep 阶段

当前 sweep 已完成训练收敛记录、retrieval downstream ablation、full classification ablation、corrected 64-sample segmentation probes，并对 K256 与 `tau_p=0.07` 完成 selected full corrected segmentation reruns。

### 7.1 Anchor count K sweep

| K | Final train loss | 分析 |
|---:|---:|---|
| 32 | 0.509355 | anchor 容量不足，loss 最高 |
| 64 | 0.403373 | 明显改善 |
| 128 | 0.338583 | 继续改善 |
| 256 | 0.303722 | 接近主模型 |
| 512 | 0.283415 | 当前主模型，loss 最低 |

结论：训练 loss 随 K 增大单调下降，retrieval Avg R@1 也随 K 单调提升（K32 `35.94` -> K512 `51.05`），符合 anchor capacity 直觉；仍需要 classification/segmentation 下游指标才能得出完整 paper-grade ablation 结论。

### 7.2 CAP temperature `tau_p` sweep

| `tau_p` | Final train loss |
|---:|---:|
| 0.02 | 0.252805 |
| 0.03 | 0.283415 |
| 0.05 | 0.335997 |
| 0.07 | 0.371047 |
| 0.10 | 0.407706 |

结论：当前训练 loss 上 `tau_p=0.02` 最低，retrieval Avg R@1 也以 `0.02` 最优（`51.41`，略高于主设置 `0.03` 的 `51.05`）；但是否替换主设置仍需 classification/segmentation 下游指标确认。

### 7.3 Token usage / pooling mode sweep

| Mode | Final train loss | 分析 |
|---|---:|---|
| global | 0.740161 | 只用 global/first token，明显最差 |
| mean | 0.501132 | 全 token mean 优于 global |
| cap | 0.283415 | CAP 最优，符合论文方向 |

结论：训练 loss 与 retrieval 都强支持 token-level CAP 的必要性；retrieval Avg R@1 为 CAP `51.05` > mean `37.26` > global `25.20`。仍需 classification/segmentation 指标完成 ablation table。

## 8. CKA proxy 与 anchor-overlap 分析

### 8.1 CKA proxy

输出：`outputs/cka/coco_karpathy_layer_sweep.json`

| Best vision layer | Best text layer | Linear CKA |
|---:|---:|---:|
| -1 | -2 | 0.665336 |

分析：当前 proxy 显示 final DINOv2 layer 与 RoBERTa 倒数第二层最接近，但这不是严格论文 CKA selection 复现。若要 paper-level layer parity，需要做 layer-specific token extraction + retraining/evaluation。

### 8.2 Anchor overlap

输出：`outputs/analysis/coco_karpathy_anchor_overlap.json`

| Metric | Value |
|---|---:|
| matched hard overlap | 0.517633 |
| matched Dice | 0.517633 |
| mismatched hard overlap | 0.436705 |
| mismatched Dice | 0.436705 |

Matched vs mismatched gap：`+0.080928` absolute，`+18.53%` relative over mismatched。

分析：PAL anchors 在匹配 image/text pairs 上确实表现出更高 overlap，支持论文关于共享相对表示结构的 qualitative claim；但 gap 不算极大，后续可补 qualitative heatmap / caption attention 可视化。

## 9. 当前最好/最新结果总表

| 模块 | 当前最好/最新结果 | Paper target | 当前状态 |
|---|---|---|---|
| 主训练 | K=512 final train loss `0.283415` | 无直接 paper target | 主模型完成 |
| COCO retrieval | Karpathy I2T/T2I R@1 `49.22 / 36.63` | `56.30 / 42.60` | paper-protocol candidate，约 86%-87% |
| Flickr30k retrieval | Karpathy I2T/T2I R@1 `67.40 / 50.96` | `76.30 / 61.80` | paper-protocol candidate，约 82%-88% |
| Classification | best prompt avg top1 `46.09` | `51.46` | 约 89.56% |
| VOC20 segmentation | corrected mIoU `20.58` | `32.30` | 63.71%，仍需 dense layer/prompt |
| Context segmentation | corrected common59 mIoU `11.23` | `25.50` | 44.05%，协议已修正但仍低 |
| ADE20K segmentation | clean aliases + `--ignore-zero` + recovered `last_hidden_state` dense tokens selected full mIoU `10.55`; diagnostic group calibration `11.47` | `13.80` | 未校准 76.45%；诊断校准 83.11%，仍需 caveat |
| K sweep | retrieval Avg R@1 `35.94 -> 51.05`；classification Avg top1 `38.20 -> 45.63` | Figure 4 qualitative curve | selected full K256 avg mIoU `19.90` |
| `tau_p` sweep | retrieval/classification 最佳 `0.02`；segmentation 最佳候选 `0.07` | ablation avg targets | selected full `tau_p=0.07` + ADE20K clean aliases / `--ignore-zero` / recovered `last_hidden_state` dense tokens avg mIoU `23.38`; diagnostic group calibration avg `23.68` |
| Token usage | CAP 在 retrieval/classification/seg probe 全部最优 | ablation avg targets | 强支持 CAP，剩余 VOC20/Context full segmentation rows 可选；ADE20K full clean rows 已完成 |
| Anchor overlap | matched `0.517633` > mismatched `0.436705` | qualitative/analysis | 方向正确 |

## 10. GitHub 保存与提交范围

建议提交源码、配置、测试与中文文档；不提交大文件产物。

应提交：

- `README.md`
- `REPRODUCTION_SUMMARY.md`
- `configs/`
- `docs/*.md`
- `scripts/*.py` 与必要 shell scripts
- `src/pal_repro/`
- `tests/`

不提交：

- `outputs/`
- `data/tokens/`
- `data/datasets/`
- `data/splits/`
- `*.pt`, `*.pth`
- `external/bridge-anchors/`
- `docs/paper_text.txt`

`.gitignore` 已覆盖上述大文件/本地产物路径。

## 11. 后续优先级

1. **ADE20K segmentation debug**：先做 16-64 sample prompt/name/layer probes，再考虑 full rerun。
2. **Context/VOC20 dense improvement**：尝试 dense-token layer selection、prompt ensemble、class-name aliases。
3. **Ablation 下游指标补齐**：retrieval 与 classification 已完成，segmentation 已有 64-sample corrected probes，并已对 K256、`tau_p=0.07` 完成 selected full corrected segmentation，ADE20K full clean-alias rows 也已补齐（见 `docs/ablation_downstream_classification_segmentation_results.md`）；下一步按需补齐剩余 VOC20/Context full segmentation rows。
4. **严格 CKA layer selection**：从论文/附录/作者实现恢复 layer indices；如无法恢复，至少用 layer sweep 给出 best/reported deviation。
5. **Qualitative visualization**：anchor heatmaps、caption word attention、segmentation examples。
6. **Baseline rows**：如目标是完整论文表格，继续实现/导入 CSA、LinearRS、MLPRS、SAIL、FA。
