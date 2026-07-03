from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.analysis import anchor_overlap_report  # noqa: E402
from pal_repro.eval import (  # noqa: E402
    evaluate_retrieval_model,
    evaluate_zero_shot_classification,
    foreground_miou,
    multicaption_retrieval_metrics,
)
from pal_repro.experiment_matrix import load_experiment_matrix, paper_target_metrics  # noqa: E402
from pal_repro.models.pal import ProjectionFreeAnchorLearning  # noqa: E402


class TestPaperEvalContract(unittest.TestCase):
    def test_model_can_encode_modalities_independently_for_eval(self) -> None:
        torch.manual_seed(0)
        model = ProjectionFreeAnchorLearning(dim_img=8, dim_txt=8, num_anchors=4)
        image_tokens = torch.randn(3, 5, 8)
        text_tokens = torch.randn(7, 6, 8)
        text_mask = torch.ones(7, 6, dtype=torch.bool)

        image_profiles = model.encode_image(image_tokens)
        text_profiles = model.encode_text(text_tokens, text_mask)

        self.assertEqual(tuple(image_profiles.shape), (3, 4))
        self.assertEqual(tuple(text_profiles.shape), (7, 4))
        self.assertTrue(torch.allclose(image_profiles.norm(dim=-1), torch.ones(3), atol=1e-5))
        self.assertTrue(torch.allclose(text_profiles.norm(dim=-1), torch.ones(7), atol=1e-5))

    def test_zero_shot_classification_uses_class_text_tokens(self) -> None:
        torch.manual_seed(1)
        model = ProjectionFreeAnchorLearning(dim_img=8, dim_txt=8, num_anchors=4)
        image_tokens = torch.randn(5, 4, 8)
        class_text_tokens = torch.randn(3, 6, 8)
        class_text_mask = torch.ones(3, 6, dtype=torch.bool)
        labels = torch.tensor([0, 1, 2, 1, 0])

        metrics = evaluate_zero_shot_classification(
            model,
            image_tokens,
            class_text_tokens,
            class_text_mask,
            labels,
            batch_size=2,
        )

        self.assertIn("top1", metrics)
        self.assertIn("top5", metrics)
        self.assertGreaterEqual(metrics["top1"], 0.0)
        self.assertLessEqual(metrics["top1"], 100.0)

    def test_retrieval_model_eval_reports_percentages(self) -> None:
        torch.manual_seed(2)
        model = ProjectionFreeAnchorLearning(dim_img=8, dim_txt=8, num_anchors=4)
        image_tokens = torch.randn(6, 4, 8)
        text_tokens = torch.randn(6, 5, 8)
        text_mask = torch.ones(6, 5, dtype=torch.bool)

        metrics = evaluate_retrieval_model(model, image_tokens, text_tokens, text_mask, batch_size=3)

        for key in ["i2t_r1", "i2t_r5", "t2i_r1", "t2i_r5", "mean_recall"]:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 100.0)

    def test_multicaption_retrieval_accepts_multiple_positive_captions(self) -> None:
        image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text_features = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]
        )
        image_ids = [101, 202]
        text_image_ids = [101, 101, 202, 202]

        metrics = multicaption_retrieval_metrics(
            image_features,
            text_features,
            image_ids=image_ids,
            text_image_ids=text_image_ids,
            ks=(1, 2),
            percent=True,
        )

        self.assertEqual(metrics["i2t_r1"], 100.0)
        self.assertEqual(metrics["t2i_r1"], 100.0)
        self.assertEqual(metrics["mean_recall"], 100.0)

    def test_foreground_miou_ignores_background_and_missing_classes(self) -> None:
        pred = torch.tensor([[0, 1, 1], [2, 2, 0]])
        target = torch.tensor([[0, 1, 2], [2, 0, 0]])
        miou = foreground_miou(pred, target, class_ids=[1, 2], background_id=0)
        self.assertAlmostEqual(miou, ((1 / 2) + (1 / 3)) / 2 * 100.0)

    def test_anchor_overlap_report_matches_paper_metric_shape(self) -> None:
        image_profiles = torch.tensor(
            [[0.9, 0.8, 0.1, 0.0], [0.1, 0.2, 0.8, 0.7], [0.6, 0.5, 0.4, 0.3]]
        )
        text_profiles = torch.tensor(
            [[0.7, 0.6, 0.2, 0.1], [0.0, 0.1, 0.9, 0.8], [0.5, 0.4, 0.7, 0.6]]
        )
        report = anchor_overlap_report(image_profiles, text_profiles, k=2)
        self.assertIn("matched_hard_overlap", report)
        self.assertIn("mismatched_hard_overlap", report)
        self.assertIn("matched_dice", report)
        self.assertIn("mismatched_dice", report)
        for value in report.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_paper_experiment_matrix_contains_all_claimed_experiments(self) -> None:
        matrix = load_experiment_matrix(ROOT / "configs" / "reproduction_matrix.yaml")
        targets = paper_target_metrics()
        self.assertIn("main_pal_k512", matrix["experiments"])
        for name in ["STL10", "CIFAR100", "Caltech101", "DTD", "EuroSAT"]:
            self.assertIn(name, targets["classification_top1"])
        for name in ["Flickr30k", "COCO"]:
            self.assertIn(name, targets["retrieval_r1"])
        for name in ["VOC20", "Context", "ADE20K"]:
            self.assertIn(name, targets["segmentation_miou_fg"])
        self.assertIn("anchor_count", matrix["ablations"])
        self.assertIn("pool_temperature", matrix["ablations"])


if __name__ == "__main__":
    unittest.main()
