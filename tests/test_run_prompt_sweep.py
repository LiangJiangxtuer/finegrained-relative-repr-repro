from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_prompt_sweep_module():
    spec = importlib.util.spec_from_file_location(
        "run_prompt_sweep",
        ROOT / "scripts" / "run_prompt_sweep.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestRunPromptSweepCli(unittest.TestCase):
    def test_parser_accepts_fixed_ensemble_and_layer_overrides(self) -> None:
        module = load_prompt_sweep_module()
        args = module.build_parser().parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--output-dir",
                "outputs/classification_prompt_ensemble",
                "--template",
                "a photo of {class_name}",
                "--template",
                "a clean photo of {class_name}",
                "--ensemble",
                "--vision-layer",
                "-6",
                "--text-layer",
                "-8",
            ]
        )

        self.assertTrue(args.ensemble)
        self.assertEqual(args.vision_layer, -6)
        self.assertEqual(args.text_layer, -8)
        self.assertEqual(
            args.template,
            ["a photo of {class_name}", "a clean photo of {class_name}"],
        )


if __name__ == "__main__":
    unittest.main()
