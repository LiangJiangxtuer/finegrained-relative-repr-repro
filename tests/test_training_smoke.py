from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.data import TokenTensorDataset, load_token_tensors, split_indices  # noqa: E402
from pal_repro.eval import recall_at_k, retrieval_metrics  # noqa: E402
from pal_repro.train import TrainConfig, train_pal  # noqa: E402


class TestTrainingAndEval(unittest.TestCase):
    def test_split_indices_is_deterministic_and_complete(self) -> None:
        train_a, eval_a = split_indices(10, train_size=6, seed=123)
        train_b, eval_b = split_indices(10, train_size=6, seed=123)
        self.assertTrue(torch.equal(train_a, train_b))
        self.assertTrue(torch.equal(eval_a, eval_b))
        self.assertEqual(len(train_a), 6)
        self.assertEqual(len(eval_a), 4)
        self.assertEqual(set(train_a.tolist()).isdisjoint(set(eval_a.tolist())), True)
        self.assertEqual(sorted(train_a.tolist() + eval_a.tolist()), list(range(10)))

    def test_token_tensor_dataset_loads_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = torch.randn(5, 3, 4).half()
            text = torch.randn(5, 2, 6).half()
            mask = torch.ones(5, 2, dtype=torch.bool)
            torch.save(image, root / "image_tokens.pt")
            torch.save(text, root / "text_tokens.pt")
            torch.save(mask, root / "text_mask.pt")

            tensors = load_token_tensors(root)
            dataset = TokenTensorDataset(tensors, indices=torch.tensor([3, 1]))

            self.assertEqual(tensors.image_tokens.dtype, torch.float16)
            self.assertEqual(tensors.text_tokens.dtype, torch.float16)
            self.assertEqual(len(dataset), 2)
            img0, txt0, mask0 = dataset[0]
            self.assertTrue(torch.equal(img0, image[3]))
            self.assertTrue(torch.equal(txt0, text[3]))
            self.assertTrue(torch.equal(mask0, mask[3]))

    def test_recall_metrics_for_identity_similarity(self) -> None:
        features = torch.eye(6)
        metrics = retrieval_metrics(features, features, ks=(1, 5))
        self.assertEqual(metrics["i2t_r1"], 1.0)
        self.assertEqual(metrics["t2i_r1"], 1.0)
        self.assertEqual(recall_at_k(features @ features.T, k=1), 1.0)

    def test_tiny_synthetic_training_writes_metrics_and_checkpoint(self) -> None:
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Correlated synthetic paired tokens so the pipeline has a real signal.
            base = torch.randn(24, 4, 8)
            image = base + 0.01 * torch.randn_like(base)
            text = base + 0.01 * torch.randn_like(base)
            mask = torch.ones(24, 4, dtype=torch.bool)
            torch.save(image, root / "image_tokens.pt")
            torch.save(text, root / "text_tokens.pt")
            torch.save(mask, root / "text_mask.pt")
            out = root / "out"

            cfg = TrainConfig(
                data_dir=root,
                output_dir=out,
                num_anchors=8,
                pool_temperature=0.03,
                contrastive_temperature=0.07,
                epochs=1,
                batch_size=8,
                lr=1e-2,
                weight_decay=0.0,
                train_size=16,
                seed=11,
                device="cpu",
            )
            result = train_pal(cfg)

            self.assertTrue((out / "checkpoint.pt").exists())
            self.assertTrue((out / "metrics.json").exists())
            metrics = json.loads((out / "metrics.json").read_text())
            self.assertEqual(metrics["train_size"], 16)
            self.assertEqual(metrics["eval_size"], 8)
            self.assertIn("eval", metrics)
            self.assertIn("i2t_r1", metrics["eval"])
            self.assertTrue(torch.isfinite(torch.tensor(result["final_train_loss"])).item())

    def test_training_can_use_all_samples_without_internal_eval_split(self) -> None:
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = torch.randn(8, 3, 6).half()
            torch.save(base, root / "image_tokens.pt")
            torch.save(base, root / "text_tokens.pt")
            torch.save(torch.ones(8, 3, dtype=torch.bool), root / "text_mask.pt")
            out = root / "full_train"

            result = train_pal(
                TrainConfig(
                    data_dir=root,
                    output_dir=out,
                    num_anchors=4,
                    epochs=1,
                    batch_size=4,
                    train_size=8,
                    device="cpu",
                )
            )

            metrics = json.loads((out / "metrics.json").read_text())
            self.assertEqual(metrics["train_size"], 8)
            self.assertEqual(metrics["eval_size"], 0)
            self.assertEqual(metrics["eval"], {})
            self.assertTrue(torch.isfinite(torch.tensor(result["final_train_loss"])).item())


if __name__ == "__main__":
    unittest.main()
