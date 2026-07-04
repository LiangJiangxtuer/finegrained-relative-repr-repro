from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.losses import symmetric_info_nce_loss  # noqa: E402
from pal_repro.models.pal import (  # noqa: E402
    PALOutput,
    ProjectionFreeAnchorLearning,
    pal_trainable_parameter_names,
)


class TestPALCore(unittest.TestCase):
    def test_forward_shapes_normalization_and_token_sims(self) -> None:
        torch.manual_seed(0)
        model = ProjectionFreeAnchorLearning(
            dim_img=8,
            dim_txt=10,
            num_anchors=4,
            pool_temperature=0.03,
        )
        img_tokens = torch.randn(2, 5, 8)
        txt_tokens = torch.randn(2, 7, 10)
        txt_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        output = model(img_tokens, txt_tokens, txt_mask, return_token_sims=True)

        self.assertIsInstance(output, PALOutput)
        self.assertEqual(tuple(output.image.shape), (2, 4))
        self.assertEqual(tuple(output.text.shape), (2, 4))
        self.assertIsNotNone(output.image_token_sims)
        self.assertIsNotNone(output.text_token_sims)
        self.assertEqual(tuple(output.image_token_sims.shape), (2, 5, 4))
        self.assertEqual(tuple(output.text_token_sims.shape), (2, 7, 4))
        self.assertTrue(torch.allclose(output.image.norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(output.text.norm(dim=-1), torch.ones(2), atol=1e-5))

    def test_k512_has_anchor_only_trainable_parameters(self) -> None:
        model = ProjectionFreeAnchorLearning(
            dim_img=1024,
            dim_txt=1024,
            num_anchors=512,
            pool_temperature=0.03,
        )

        self.assertEqual(tuple(model.anchors_img.shape), (512, 1024))
        self.assertEqual(tuple(model.anchors_txt.shape), (512, 1024))
        self.assertEqual(pal_trainable_parameter_names(model), ["anchors_img", "anchors_txt"])

    def test_symmetric_infonce_backpropagates_to_anchors(self) -> None:
        torch.manual_seed(1)
        model = ProjectionFreeAnchorLearning(dim_img=8, dim_txt=8, num_anchors=4)
        output = model(
            img_tokens=torch.randn(3, 5, 8),
            txt_tokens=torch.randn(3, 6, 8),
            txt_mask=torch.ones(3, 6, dtype=torch.bool),
        )

        loss = symmetric_info_nce_loss(output.image, output.text, temperature=0.07)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.anchors_img.grad)
        self.assertIsNotNone(model.anchors_txt.grad)

    def test_text_mask_requires_at_least_one_valid_token(self) -> None:
        model = ProjectionFreeAnchorLearning(dim_img=4, dim_txt=4, num_anchors=2)
        with self.assertRaisesRegex(ValueError, "at least one valid token"):
            model(
                img_tokens=torch.randn(1, 3, 4),
                txt_tokens=torch.randn(1, 3, 4),
                txt_mask=torch.zeros(1, 3, dtype=torch.bool),
            )

    def test_mean_pooling_uses_valid_token_average_for_ablation(self) -> None:
        model = ProjectionFreeAnchorLearning(dim_img=3, dim_txt=3, num_anchors=2, pooling_mode="mean")
        with torch.no_grad():
            model.anchors_img.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
            model.anchors_txt.copy_(model.anchors_img)
        txt_tokens = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        txt_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

        text_profile = model.encode_text(txt_tokens, txt_mask)
        expected = torch.nn.functional.normalize(torch.tensor([[0.5, 0.5]]), dim=-1)

        self.assertTrue(torch.allclose(text_profile, expected, atol=1e-6))

    def test_global_pooling_uses_first_token_for_ablation(self) -> None:
        model = ProjectionFreeAnchorLearning(dim_img=3, dim_txt=3, num_anchors=2, pooling_mode="global")
        with torch.no_grad():
            model.anchors_img.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        img_tokens = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])

        image_profile = model.encode_image(img_tokens)

        self.assertTrue(torch.allclose(image_profile, torch.tensor([[0.0, 1.0]]), atol=1e-6))

    def test_input_dimension_validation(self) -> None:
        model = ProjectionFreeAnchorLearning(dim_img=4, dim_txt=4, num_anchors=2)
        with self.assertRaisesRegex(ValueError, "img_tokens"):
            model(torch.randn(2, 4), torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
