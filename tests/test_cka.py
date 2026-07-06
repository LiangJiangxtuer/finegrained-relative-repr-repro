from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.cka import linear_cka, rank_layer_pairs  # noqa: E402


def load_cka_sweep_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_cka_layer_sweep",
        ROOT / "scripts" / "run_cka_layer_sweep.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestCKAUtilities(unittest.TestCase):
    def test_linear_cka_is_one_for_identical_features(self) -> None:
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=6)

    def test_rank_layer_pairs_sorts_scores_descending(self) -> None:
        vision = {0: torch.eye(3), 1: torch.ones(3, 3)}
        text = {0: torch.eye(3), 1: torch.randn(3, 3)}

        ranked = rank_layer_pairs(vision, text)

        self.assertEqual(ranked[0]["vision_layer"], 0)
        self.assertEqual(ranked[0]["text_layer"], 0)
        self.assertGreaterEqual(ranked[0]["cka"], ranked[-1]["cka"])

    def test_cka_sweep_defaults_to_paper_parity_grid_and_1024_pairs(self) -> None:
        module = load_cka_sweep_module()
        args = module.finalize_args(
            module.build_parser().parse_args(
                [
                    "--dataset",
                    "coco",
                    "--karpathy-json",
                    "dataset_coco.json",
                    "--coco-root",
                    "coco",
                    "--output",
                    "outputs/cka/coco_karpathy_layer_sweep.json",
                ]
            )
        )

        self.assertEqual(args.limit_images, 1024)
        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.vision_layer, module.DEFAULT_LAYER_GRID)
        self.assertEqual(args.text_layer, module.DEFAULT_LAYER_GRID)

    def test_candidate_layer_pairs_keep_top_k_plus_final_layer_baseline(self) -> None:
        module = load_cka_sweep_module()
        ranked = [
            {"vision_layer": -2, "text_layer": -2, "cka": 0.9},
            {"vision_layer": -4, "text_layer": -4, "cka": 0.8},
            {"vision_layer": -6, "text_layer": -6, "cka": 0.7},
            {"vision_layer": -1, "text_layer": -1, "cka": 0.6},
        ]

        selected = module.select_candidate_layer_pairs(ranked, top_k=3)

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [(row["vision_layer"], row["text_layer"]) for row in selected],
            [(-2, -2), (-4, -4), (-6, -6), (-1, -1)],
        )
        self.assertEqual(selected[-1]["selection_reason"], "final_layer_baseline")


if __name__ == "__main__":
    unittest.main()
