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
        self.assertLess(names.index("eval_flickr_karpathy_test"), names.index("classification_prompt_ensemble"))
        self.assertLess(names.index("voc20_full_segmentation"), names.index("train_k32"))
        self.assertLess(names.index("train_k512_tau_0_02"), names.index("anchor_overlap_analysis"))

    def test_pipeline_commands_include_required_outputs_and_tau_override(self) -> None:
        steps = {step.name: step for step in build_pipeline_steps(PipelineConfig(root=Path("/repo")))}

        coco = steps["extract_coco_karpathy_test"]
        self.assertIn("extract_karpathy_retrieval_tokens.py", " ".join(coco.command))
        self.assertIn("coco2014_karpathy_test_multicaption", " ".join(coco.command))

        tau = steps["train_k512_tau_0_02"]
        self.assertIn("--pool-temperature", tau.command)
        self.assertIn("0.02", tau.command)
        self.assertEqual(tau.output, Path("/repo/outputs/ablations/tau_0_02/metrics.json"))

        mean = steps["train_token_usage_mean"]
        self.assertIn("--pooling-mode", mean.command)
        self.assertIn("mean", mean.command)
        self.assertEqual(mean.output, Path("/repo/outputs/ablations/token_usage_mean/metrics.json"))

    def test_pipeline_segmentation_steps_use_corrected_paper_protocols(self) -> None:
        steps = {step.name: step for step in build_pipeline_steps(PipelineConfig(root=Path("/repo")))}

        voc = steps["voc20_full_segmentation"]
        self.assertIn("--target-frame", voc.command)
        self.assertIn("processor", voc.command)

        context = steps["context_full_segmentation"]
        self.assertIn("--target-frame", context.command)
        self.assertIn("processor", context.command)
        self.assertIn("--context-protocol", context.command)
        self.assertIn("common59", context.command)

        ade = steps["ade20k_full_segmentation"]
        self.assertIn("--target-frame", ade.command)
        self.assertIn("processor", ade.command)

    def test_pipeline_classification_uses_fixed_prompt_ensemble(self) -> None:
        steps = {step.name: step for step in build_pipeline_steps(PipelineConfig(root=Path("/repo")))}

        classification = steps["classification_prompt_ensemble"]
        self.assertIn("--ensemble", classification.command)
        self.assertEqual(classification.command.count("--template"), 4)
        joined = " ".join(classification.command)
        self.assertIn("a photo of {class_name}", joined)
        self.assertIn("a close-up photo of {class_name}", joined)
        self.assertIn("a cropped photo of {class_name}", joined)
        self.assertIn("a clean photo of {class_name}", joined)
        self.assertEqual(classification.output, Path("/repo/outputs/classification_prompt_ensemble/summary.json"))

    def test_pipeline_cka_step_uses_paper_parity_grid(self) -> None:
        steps = {step.name: step for step in build_pipeline_steps(PipelineConfig(root=Path("/repo")))}

        cka = steps["cka_layer_sweep_proxy"]
        self.assertIn("--limit-images", cka.command)
        self.assertIn("1024", cka.command)
        self.assertIn("--top-k", cka.command)
        self.assertIn("3", cka.command)
        self.assertEqual(cka.command.count("--vision-layer"), 10)
        self.assertEqual(cka.command.count("--text-layer"), 10)
        for layer in ("-1", "-2", "-4", "-6", "-8", "-10", "-12", "-16", "-20", "-24"):
            self.assertIn(layer, cka.command)

    def test_pipeline_passes_layer_overrides_to_extraction_classification_and_segmentation(self) -> None:
        config = PipelineConfig(root=Path("/repo"), vision_layer=-6, text_layer=-8)
        steps = {step.name: step for step in build_pipeline_steps(config)}

        for name in (
            "extract_coco_karpathy_test",
            "extract_flickr_karpathy_test",
            "classification_prompt_ensemble",
            "voc20_full_segmentation",
            "context_full_segmentation",
            "ade20k_full_segmentation",
        ):
            command = steps[name].command
            self.assertIn("--vision-layer", command, name)
            self.assertIn("-6", command, name)
            self.assertIn("--text-layer", command, name)
            self.assertIn("-8", command, name)


if __name__ == "__main__":
    unittest.main()
