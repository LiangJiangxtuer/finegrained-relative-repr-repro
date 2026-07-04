"""Ordered execution plan for the remaining PAL paper-reproduction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineStep:
    """One executable reproduction pipeline step."""

    name: str
    priority: int
    command: list[str]
    output: Path | None = None
    description: str = ""
    gpu: bool = False
    long_running: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    """Paths and defaults used to construct the paper-grade pipeline."""

    root: Path
    python: Path = Path("/home/hnxxzy/miniconda3/envs/ovvs/bin/python")
    checkpoint: Path | None = None
    coco_root: Path = Path("/home/hnxxzy/projects/DeepScientist/quests/pal-relative-rep-repro/tmp/datasets/coco2014/raw")
    flickr_zip: Path = Path("/home/hnxxzy/Downloads/Flickr30k.zip")
    batch_size_extract: int = 8
    batch_size_eval: int = 256
    train_batch_size: int = 128
    train_epochs: int = 20
    train_size: int = 82783
    prompt_templates: tuple[str, ...] = field(
        default=(
            "a photo of {class_name}",
            "a cropped photo of {class_name}",
            "a close-up photo of {class_name}",
            "a clean photo of {class_name}",
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        if self.checkpoint is None:
            object.__setattr__(
                self,
                "checkpoint",
                self.root / "outputs/pal_k512_coco2014_full/checkpoint.pt",
            )


def _py(config: PipelineConfig, *parts: str | Path) -> list[str]:
    return [str(config.python), *[str(part) for part in parts]]


def _module(config: PipelineConfig, module: str, *args: str | Path) -> list[str]:
    return [str(config.python), "-m", module, *[str(arg) for arg in args]]


def build_pipeline_steps(config: PipelineConfig) -> list[PipelineStep]:
    """Return all remaining reproduction steps in priority order.

    The first steps align retrieval splits and run retrieval again. Later steps
    perform prompt/segmentation sweeps, full anchor ablations, analysis, and
    baseline placeholders. Expensive stages are marked as long-running so the
    top-level runner can be launched under Hermes background process tracking.
    """

    root = config.root
    checkpoint = config.checkpoint or root / "outputs/pal_k512_coco2014_full/checkpoint.pt"
    split_dir = root / "data/splits/karpathy"
    coco_json = split_dir / "dataset_coco.json"
    flickr_json = split_dir / "dataset_flickr30k.json"
    coco_official_tokens = root / "data/tokens/coco2014_karpathy_test_multicaption"
    flickr_official_tokens = root / "data/tokens/flickr30k_karpathy_test_multicaption"

    steps: list[PipelineStep] = [
        PipelineStep(
            name="download_karpathy_splits",
            priority=10,
            command=_py(config, root / "scripts/prepare_karpathy_splits.py", "--output-dir", split_dir),
            output=flickr_json,
            description="Download/extract Karpathy COCO/Flickr30k split metadata.",
        ),
        PipelineStep(
            name="extract_coco_karpathy_test",
            priority=20,
            command=_py(
                config,
                root / "scripts/extract_karpathy_retrieval_tokens.py",
                "--dataset", "coco",
                "--karpathy-json", coco_json,
                "--split", "test",
                "--coco-root", config.coco_root,
                "--output-dir", coco_official_tokens,
                "--caption-policy", "all",
                "--batch-size", str(config.batch_size_extract),
                "--chunk-size", "2048",
                "--local-files-only",
            ),
            output=coco_official_tokens / "metadata.json",
            description="Extract official Karpathy COCO test multi-caption tokens.",
            gpu=True,
            long_running=True,
        ),
        PipelineStep(
            name="eval_coco_karpathy_test",
            priority=30,
            command=_module(
                config,
                "pal_repro.evaluate",
                "retrieval-multicaption",
                "--checkpoint", checkpoint,
                "--token-dir", coco_official_tokens,
                "--output", root / "outputs/pal_k512_coco2014_full/coco_karpathy_test_multicaption_retrieval.json",
                "--batch-size", str(config.batch_size_eval),
            ),
            output=root / "outputs/pal_k512_coco2014_full/coco_karpathy_test_multicaption_retrieval.json",
            description="Evaluate COCO retrieval on official Karpathy test split.",
            gpu=True,
        ),
        PipelineStep(
            name="extract_flickr_karpathy_test",
            priority=40,
            command=_py(
                config,
                root / "scripts/extract_karpathy_retrieval_tokens.py",
                "--dataset", "flickr30k",
                "--karpathy-json", flickr_json,
                "--split", "test",
                "--flickr-zip", config.flickr_zip,
                "--output-dir", flickr_official_tokens,
                "--caption-policy", "all",
                "--batch-size", str(config.batch_size_extract),
                "--chunk-size", "2048",
                "--local-files-only",
            ),
            output=flickr_official_tokens / "metadata.json",
            description="Extract official Karpathy Flickr30k test multi-caption tokens.",
            gpu=True,
            long_running=True,
        ),
        PipelineStep(
            name="eval_flickr_karpathy_test",
            priority=50,
            command=_module(
                config,
                "pal_repro.evaluate",
                "retrieval-multicaption",
                "--checkpoint", checkpoint,
                "--token-dir", flickr_official_tokens,
                "--output", root / "outputs/pal_k512_coco2014_full/flickr30k_karpathy_test_multicaption_retrieval.json",
                "--batch-size", str(config.batch_size_eval),
            ),
            output=root / "outputs/pal_k512_coco2014_full/flickr30k_karpathy_test_multicaption_retrieval.json",
            description="Evaluate Flickr30k retrieval on official Karpathy test split.",
            gpu=True,
        ),
        PipelineStep(
            name="cka_layer_sweep_proxy",
            priority=55,
            command=_py(
                config,
                root / "scripts/run_cka_layer_sweep.py",
                "--dataset", "coco",
                "--karpathy-json", coco_json,
                "--split", "test",
                "--coco-root", config.coco_root,
                "--caption-policy", "first",
                "--limit-images", "128",
                "--output", root / "outputs/cka/coco_karpathy_layer_sweep.json",
                "--local-files-only",
            ),
            output=root / "outputs/cka/coco_karpathy_layer_sweep.json",
            description="Run a CKA proxy sweep to rank DINOv2/RoBERTa layer pairs before full layer-specific retraining.",
            gpu=True,
            long_running=True,
        ),
        PipelineStep(
            name="prompt_sweep_classification",
            priority=60,
            command=_py(
                config,
                root / "scripts/run_prompt_sweep.py",
                "--checkpoint", checkpoint,
                "--output-dir", root / "outputs/prompt_sweep/classification",
                "--batch-size", "64",
                *sum((["--template", template] for template in config.prompt_templates), []),
            ),
            output=root / "outputs/prompt_sweep/classification/summary.json",
            description="Run classification prompt-template sweep.",
            gpu=True,
            long_running=True,
        ),
        PipelineStep(
            name="voc20_full_segmentation",
            priority=70,
            command=_py(
                config,
                root / "scripts/evaluate_segmentation.py",
                "--dataset", "VOC20",
                "--checkpoint", checkpoint,
                "--output", root / "outputs/pal_k512_coco2014_full/voc20_segmentation_full.json",
                "--batch-size", "8",
                "--local-files-only",
            ),
            output=root / "outputs/pal_k512_coco2014_full/voc20_segmentation_full.json",
            description="Run full VOC20 foreground-mIoU evaluation.",
            gpu=True,
            long_running=True,
        ),
        PipelineStep(
            name="context_full_segmentation",
            priority=80,
            command=_py(
                config,
                root / "scripts/evaluate_segmentation.py",
                "--dataset", "Context",
                "--checkpoint", checkpoint,
                "--output", root / "outputs/pal_k512_coco2014_full/context_segmentation_full.json",
                "--batch-size", "8",
                "--local-files-only",
            ),
            output=root / "outputs/pal_k512_coco2014_full/context_segmentation_full.json",
            description="Run Pascal Context foreground-mIoU evaluation.",
            gpu=True,
            long_running=True,
        ),
        PipelineStep(
            name="ade20k_full_segmentation",
            priority=90,
            command=_py(
                config,
                root / "scripts/evaluate_segmentation.py",
                "--dataset", "ADE20K",
                "--checkpoint", checkpoint,
                "--output", root / "outputs/pal_k512_coco2014_full/ade20k_segmentation_full.json",
                "--batch-size", "8",
                "--local-files-only",
            ),
            output=root / "outputs/pal_k512_coco2014_full/ade20k_segmentation_full.json",
            description="Run ADE20K foreground-mIoU evaluation.",
            gpu=True,
            long_running=True,
        ),
    ]

    for k in (32, 64, 128, 256, 512):
        out = root / f"outputs/ablations/k_{k}"
        steps.append(
            PipelineStep(
                name=f"train_k{k}",
                priority=100 + k,
                command=_module(
                    config,
                    "pal_repro.train",
                    "--config", root / "configs/pal_strict.yaml",
                    "--data-dir", root / "data/tokens/coco2014_full",
                    "--output-dir", out,
                    "--num-anchors", str(k),
                    "--epochs", str(config.train_epochs),
                    "--batch-size", str(config.train_batch_size),
                    "--train-size", str(config.train_size),
                ),
                output=out / "metrics.json",
                description=f"Train anchor-count ablation K={k}.",
                gpu=True,
                long_running=True,
            )
        )

    for label, tau in (("0_02", "0.02"), ("0_05", "0.05"), ("0_07", "0.07"), ("0_10", "0.10")):
        out = root / f"outputs/ablations/tau_{label}"
        steps.append(
            PipelineStep(
                name=f"train_k512_tau_{label}",
                priority=700 + int(float(tau) * 1000),
                command=_module(
                    config,
                    "pal_repro.train",
                    "--config", root / "configs/pal_strict.yaml",
                    "--data-dir", root / "data/tokens/coco2014_full",
                    "--output-dir", out,
                    "--num-anchors", "512",
                    "--pool-temperature", tau,
                    "--epochs", str(config.train_epochs),
                    "--batch-size", str(config.train_batch_size),
                    "--train-size", str(config.train_size),
                ),
                output=out / "metrics.json",
                description=f"Train CAP-temperature ablation tau_p={tau}.",
                gpu=True,
                long_running=True,
            )
        )

    for idx, mode in enumerate(("global", "mean", "cap")):
        out = root / f"outputs/ablations/token_usage_{mode}"
        steps.append(
            PipelineStep(
                name=f"train_token_usage_{mode}",
                priority=820 + idx,
                command=_module(
                    config,
                    "pal_repro.train",
                    "--config", root / "configs/pal_strict.yaml",
                    "--data-dir", root / "data/tokens/coco2014_full",
                    "--output-dir", out,
                    "--num-anchors", "512",
                    "--pool-temperature", "0.03",
                    "--pooling-mode", mode,
                    "--epochs", str(config.train_epochs),
                    "--batch-size", str(config.train_batch_size),
                    "--train-size", str(config.train_size),
                ),
                output=out / "metrics.json",
                description=f"Train token-usage ablation with pooling_mode={mode}.",
                gpu=True,
                long_running=True,
            )
        )

    steps.extend(
        [
            PipelineStep(
                name="anchor_overlap_analysis",
                priority=900,
                command=_py(
                    config,
                    root / "scripts/analyze_anchor_overlap.py",
                    "--checkpoint", checkpoint,
                    "--token-dir", coco_official_tokens,
                    "--output", root / "outputs/analysis/coco_karpathy_anchor_overlap.json",
                    "--batch-size", str(config.batch_size_eval),
                ),
                output=root / "outputs/analysis/coco_karpathy_anchor_overlap.json",
                description="Compute Table-4-style top-k anchor overlap/Dice on COCO.",
                gpu=True,
            ),
            PipelineStep(
                name="baseline_tracking_report",
                priority=1000,
                command=_py(
                    config,
                    root / "scripts/write_baseline_tracking_report.py",
                    "--output", root / "outputs/reports/baseline_tracking.json",
                ),
                output=root / "outputs/reports/baseline_tracking.json",
                description="Record baseline reproduction status and required external implementations.",
            ),
        ]
    )
    return sorted(steps, key=lambda step: step.priority)
