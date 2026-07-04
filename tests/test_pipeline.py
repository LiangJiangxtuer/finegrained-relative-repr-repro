from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.pipeline import PipelineConfig, build_pipeline_steps  # noqa: E402


class TestReproductionPipeline(unittest.TestCase):
    def test_pipeline_orders_official_split_before_sweeps_and_ablations(self) -> None:
        steps = build_pipeline_steps(PipelineConfig(root=Path("/repo")))
        names = [step.name for step in steps]

        self.assertLess(names.index("download_karpathy_splits"), names.index("extract_coco_karpathy_test"))
        self.assertLess(names.index("eval_flickr_karpathy_test"), names.index("prompt_sweep_classification"))
        self.assertLess(names.index("voc20_full_segmentation"), names.index("train_k32"))
        self.assertLess(names.index("train_k512_tau_0_02"), names.index("anchor_overlap_analysis"))

    def test_pipeline_commands_include_required_outputs_and_tau_override(self) -> None:
        steps = {step.name: step for step in build_pipeline_steps(PipelineConfig(root=Path("/repo")))}

        coco = steps["extract_coco_karpathy_test"]
        self.assertIn("extract_karpathy_retrieval_tokens.py", " ".join(coco.command))
        self.assertIn("data/tokens/coco2014_karpathy_test_multicaption", " ".join(coco.command))

        tau = steps["train_k512_tau_0_02"]
        self.assertIn("--pool-temperature", tau.command)
        self.assertIn("0.02", tau.command)
        self.assertEqual(tau.output, Path("/repo/outputs/ablations/tau_0_02/metrics.json"))

        mean = steps["train_token_usage_mean"]
        self.assertIn("--pooling-mode", mean.command)
        self.assertIn("mean", mean.command)
        self.assertEqual(mean.output, Path("/repo/outputs/ablations/token_usage_mean/metrics.json"))


if __name__ == "__main__":
    unittest.main()
