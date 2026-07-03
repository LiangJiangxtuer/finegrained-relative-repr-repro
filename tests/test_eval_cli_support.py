from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.evaluate import (  # noqa: E402
    evaluate_multicaption_retrieval_token_dir,
    evaluate_retrieval_token_dir,
    load_trained_pal_model,
)
from pal_repro.models.pal import ProjectionFreeAnchorLearning  # noqa: E402


class TestEvalCliSupport(unittest.TestCase):
    def test_load_trained_pal_model_restores_anchor_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = ProjectionFreeAnchorLearning(dim_img=6, dim_txt=6, num_anchors=4)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "dim_img": 6,
                "dim_txt": 6,
                "config": {"num_anchors": 4, "pool_temperature": 0.03},
                "parameter_names": ["anchors_img", "anchors_txt"],
            }
            ckpt_path = root / "checkpoint.pt"
            torch.save(checkpoint, ckpt_path)

            loaded = load_trained_pal_model(ckpt_path, device="cpu")

            self.assertEqual(loaded.num_anchors, 4)
            self.assertEqual(loaded.dim_img, 6)
            self.assertEqual(loaded.dim_txt, 6)
            self.assertTrue(torch.allclose(loaded.anchors_img, model.anchors_img))

    def test_evaluate_retrieval_token_dir_writes_json_metrics(self) -> None:
        torch.manual_seed(21)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_dir = root / "tokens"
            token_dir.mkdir()
            base = torch.randn(5, 3, 6).half()
            torch.save(base, token_dir / "image_tokens.pt")
            torch.save(base, token_dir / "text_tokens.pt")
            torch.save(torch.ones(5, 3, dtype=torch.bool), token_dir / "text_mask.pt")

            model = ProjectionFreeAnchorLearning(dim_img=6, dim_txt=6, num_anchors=4)
            ckpt_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dim_img": 6,
                    "dim_txt": 6,
                    "config": {"num_anchors": 4, "pool_temperature": 0.03},
                    "parameter_names": ["anchors_img", "anchors_txt"],
                },
                ckpt_path,
            )
            output = root / "retrieval_metrics.json"

            metrics = evaluate_retrieval_token_dir(
                checkpoint_path=ckpt_path,
                token_dir=token_dir,
                output_path=output,
                batch_size=2,
                device="cpu",
            )

            self.assertTrue(output.exists())
            saved = json.loads(output.read_text())
            self.assertEqual(saved["num_samples"], 5)
            self.assertIn("i2t_r1", saved["metrics"])
            self.assertIn("mean_recall", metrics["metrics"])
    def test_evaluate_retrieval_chunked_token_dir_writes_json_metrics(self) -> None:
        torch.manual_seed(22)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_dir = root / "chunked_tokens"
            chunk_dir = token_dir / "chunks"
            chunk_dir.mkdir(parents=True)
            chunks = []
            start = 0
            for idx, count in enumerate([3, 2]):
                base = torch.randn(count, 3, 6).half()
                image_path = chunk_dir / f"chunk_{idx:05d}_image_tokens.pt"
                text_path = chunk_dir / f"chunk_{idx:05d}_text_tokens.pt"
                mask_path = chunk_dir / f"chunk_{idx:05d}_text_mask.pt"
                torch.save(base, image_path)
                torch.save(base, text_path)
                torch.save(torch.ones(count, 3, dtype=torch.bool), mask_path)
                chunks.append(
                    {
                        "chunk_index": idx,
                        "start": start,
                        "end": start + count,
                        "num_samples": count,
                        "image_tokens": str(image_path.relative_to(token_dir)),
                        "text_tokens": str(text_path.relative_to(token_dir)),
                        "text_mask": str(mask_path.relative_to(token_dir)),
                    }
                )
                start += count
            (token_dir / "metadata.json").write_text(json.dumps({"format": "chunks", "chunks": chunks}))

            model = ProjectionFreeAnchorLearning(dim_img=6, dim_txt=6, num_anchors=4)
            ckpt_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dim_img": 6,
                    "dim_txt": 6,
                    "config": {"num_anchors": 4, "pool_temperature": 0.03},
                    "parameter_names": ["anchors_img", "anchors_txt"],
                },
                ckpt_path,
            )
            output = root / "chunked_retrieval_metrics.json"

            metrics = evaluate_retrieval_token_dir(
                checkpoint_path=ckpt_path,
                token_dir=token_dir,
                output_path=output,
                batch_size=2,
                device="cpu",
            )

            self.assertEqual(metrics["num_samples"], 5)
            self.assertEqual(metrics["token_format"], "chunks")
            self.assertIn("i2t_r1", metrics["metrics"])
    def test_evaluate_multicaption_retrieval_uses_pairs_jsonl_image_ids(self) -> None:
        torch.manual_seed(23)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token_dir = root / "multi_tokens"
            token_dir.mkdir()
            image_tokens = torch.randn(4, 3, 6).half()
            text_tokens = image_tokens.clone()
            text_mask = torch.ones(4, 3, dtype=torch.bool)
            torch.save(image_tokens, token_dir / "image_tokens.pt")
            torch.save(text_tokens, token_dir / "text_tokens.pt")
            torch.save(text_mask, token_dir / "text_mask.pt")
            rows = [
                {"image_id": 10, "annotation_id": 100, "caption_index": 0},
                {"image_id": 10, "annotation_id": 101, "caption_index": 1},
                {"image_id": 20, "annotation_id": 200, "caption_index": 0},
                {"image_id": 20, "annotation_id": 201, "caption_index": 1},
            ]
            pairs_path = token_dir / "pairs.jsonl"
            pairs_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (token_dir / "metadata.json").write_text(
                json.dumps({"format": "monolithic", "pairs_jsonl": str(pairs_path)}),
                encoding="utf-8",
            )

            model = ProjectionFreeAnchorLearning(dim_img=6, dim_txt=6, num_anchors=4)
            ckpt_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dim_img": 6,
                    "dim_txt": 6,
                    "config": {"num_anchors": 4, "pool_temperature": 0.03},
                    "parameter_names": ["anchors_img", "anchors_txt"],
                },
                ckpt_path,
            )
            output = root / "multi_retrieval.json"

            result = evaluate_multicaption_retrieval_token_dir(
                checkpoint_path=ckpt_path,
                token_dir=token_dir,
                output_path=output,
                batch_size=2,
                device="cpu",
            )

            self.assertTrue(output.exists())
            self.assertEqual(result["num_images"], 2)
            self.assertEqual(result["num_texts"], 4)
            self.assertIn("i2t_r1", result["metrics"])
            self.assertIn("t2i_r1", result["metrics"])


if __name__ == "__main__":
    unittest.main()
