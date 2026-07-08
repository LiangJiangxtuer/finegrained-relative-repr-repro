# PAL 论文复现：Learning Relative Representations for Fine-Grained Multimodal Alignment

本仓库实现论文中的 **Projection-Free Anchor Learning (PAL)** 核心路径。官方代码未公开，因此这里参考 `shiwonkim/bridge-anchors` 的 anchor/contrastive 设计，但单独实现一个严格的 PAL 模块，避免参考仓库里的 projector、router、fixed anchor、auxiliary loss 等实验性扩展混入。

## 已完成的本地发现

- 论文文本已抽取到 `docs/paper_text.txt`。
- 复现计划已保存到 `docs/reproduction_plan.md`。
- 参考实现已克隆到 `external/bridge-anchors`。
- 推荐本地 conda Python：`/home/hnxxzy/miniconda3/envs/ovvs/bin/python`。
  - 已验证：Python 3.11.15、torch 2.5.1+cu124、CUDA 可用。
  - 当前 TUI shell 下 `conda activate`/`conda run` 会触发 conda activate TypeError，因此脚本直接调用该 conda 环境的 Python 解释器。
- 本地 COCO2014 原始数据：`/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/coco2014/raw`。
- 本地 192 样本 DINOv2-L/RoBERTa-L token cache：`/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/artifacts/runs/pal_scoped_coco_retrieval_pilot_128x64/tokens`。

## 方法实现范围

核心模块：`src/pal_repro/models/pal.py`

严格匹配论文公式：

1. L2-normalize frozen image/text tokens and modality anchors.
2. Compute token-to-anchor cosine similarities.
3. Apply Cross-Attention Pooling (CAP): anchor-wise softmax over token positions with `tau_p=0.03`.
4. Weighted-sum each anchor column into a `(B,K)` profile.
5. L2-normalize image/text profiles.
6. Train with symmetric InfoNCE (`tau=0.07`).

只有两个可训练参数：`anchors_img` 和 `anchors_txt`。

## 运行测试

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m unittest discover -s tests -v
```

## 运行本地 smoke 训练

```bash
bash scripts/run_local_smoke.sh
```

等价命令：

```bash
PYTHONPATH=src /home/hnxxzy/miniconda3/envs/ovvs/bin/python -m pal_repro.train \
  --config configs/pal_strict.yaml \
  --preset smoke \
  --output-dir outputs/local_smoke
```

输出：

- `outputs/local_smoke/checkpoint.pt`
- `outputs/local_smoke/metrics.json`

## 全量复现路线

1. 用 `scripts/extract_coco_tokens.py` 或等价脚本对 COCO2014 train first-caption pairs 提取 DINOv2 ViT-L 与 RoBERTa-Large token tensors。
2. 将 `configs/pal_strict.yaml` 的 `data.token_dir` 指向全量 token cache。
3. 使用 `K=512, tau_p=0.03, tau=0.07` 训练。
4. 扩展/运行完整 evaluation contract：
   - STL10/CIFAR100/Caltech101/DTD/EuroSAT zero-shot classification top-1；
   - Flickr30k/COCO I2T/T2I retrieval R@1；
   - VOC20/Context/ADE20K foreground mIoU。

当前仓库已经完成 PAL 核心、COCO2014 全量 token extraction、K=512 主模型训练、COCO/Flickr30k Karpathy retrieval、5 个分类数据集 zero-shot 评测、VOC20/Context/ADE20K corrected segmentation rerun、K / `tau_p` / token usage 训练型 sweep、K / `tau_p` / token usage downstream retrieval ablation、full classification ablation、corrected 64-sample segmentation probes、selected full corrected segmentation reruns、ADE20K clean-alias full rows、ADE20K dense-token/layer recovery、ADE20K targeted group-calibration diagnostics、CKA proxy 和 anchor-overlap 分析。最新中文总览见 `REPRODUCTION_SUMMARY.md`、`docs/recent_experiment_audit_summary.md`、`docs/results_snapshot.md`、`docs/ablation_downstream_retrieval_results.md`、`docs/ablation_downstream_classification_segmentation_results.md`、`docs/ade20k_frequent_class_error_analysis.md`、`docs/ade20k_dense_protocol_recovery.md`、`docs/ade20k_group_calibration_results.md` 与 `docs/zh_experiment_design_results_analysis.md`。

仍待完成的 paper-grade 部分主要包括：按需补齐剩余 K / `tau_p` / token-CAP checkpoint 的 VOC20/Context full corrected segmentation rows，如需把 calibration 当作主结果则增加 held-out calibration protocol，恢复严格 CKA layer selection，并按需实现 baseline 对比实验。
