from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.classification import (  # noqa: E402
    build_class_prompts,
    classification_topk,
    normalize_class_name,
)


class TestClassificationSupport(unittest.TestCase):
    def test_normalize_class_name_handles_dataset_conventions(self) -> None:
        self.assertEqual(normalize_class_name("aquarium_fish"), "aquarium fish")
        self.assertEqual(normalize_class_name("AnnualCrop"), "annual crop")
        self.assertEqual(normalize_class_name("Faces_easy"), "faces easy")

    def test_build_class_prompts_uses_article_free_photo_template(self) -> None:
        prompts = build_class_prompts(["aquarium_fish", "AnnualCrop"])
        self.assertEqual(prompts, ["a photo of aquarium fish", "a photo of annual crop"])

    def test_classification_topk_reports_percentages(self) -> None:
        similarity = torch.tensor(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.2],
                [0.2, 0.3, 0.7],
                [0.7, 0.8, 0.1],
            ]
        )
        labels = torch.tensor([0, 1, 2, 0])

        metrics = classification_topk(similarity, labels, ks=(1, 2))

        self.assertEqual(metrics["top1"], 75.0)
        self.assertEqual(metrics["top2"], 100.0)


if __name__ == "__main__":
    unittest.main()
