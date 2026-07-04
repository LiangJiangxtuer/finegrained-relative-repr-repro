from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.cka import linear_cka, rank_layer_pairs  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
