from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_coco_tokens.py"


def load_extract_module():
    spec = importlib.util.spec_from_file_location("extract_coco_tokens_for_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCocoTokenExtractionPairs(unittest.TestCase):
    def test_select_hidden_state_uses_requested_layer_or_last_hidden_state(self) -> None:
        module = load_extract_module()
        import torch
        from types import SimpleNamespace

        outputs = SimpleNamespace(
            last_hidden_state=torch.tensor([[[9.0]]]),
            hidden_states=(
                torch.tensor([[[1.0]]]),
                torch.tensor([[[2.0]]]),
                torch.tensor([[[3.0]]]),
            ),
        )

        self.assertEqual(module.select_hidden_state(outputs, None).item(), 9.0)
        self.assertEqual(module.select_hidden_state(outputs, -2).item(), 2.0)
        self.assertEqual(module.select_hidden_state(outputs, 1).item(), 2.0)

    def test_parser_records_layer_selection_flags(self) -> None:
        module = load_extract_module()

        args = module.build_parser().parse_args(
            [
                "--captions-json", "captions.json",
                "--image-dir", "images",
                "--output-dir", "tokens",
                "--vision-layer", "-2",
                "--text-layer", "-6",
            ]
        )

        self.assertEqual(args.vision_layer, -2)
        self.assertEqual(args.text_layer, -6)

    def test_build_pairs_all_caption_policy_keeps_every_caption_for_selected_images(self) -> None:
        module = load_extract_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            for name in ["a.jpg", "b.jpg"]:
                (image_dir / name).write_bytes(b"placeholder")
            captions = {
                "images": [
                    {"id": 2, "file_name": "b.jpg"},
                    {"id": 1, "file_name": "a.jpg"},
                ],
                "annotations": [
                    {"id": 10, "image_id": 1, "caption": "a first"},
                    {"id": 11, "image_id": 1, "caption": "a second"},
                    {"id": 20, "image_id": 2, "caption": "b first"},
                    {"id": 21, "image_id": 2, "caption": "b second"},
                    {"id": 22, "image_id": 2, "caption": "b third"},
                ],
            }
            captions_path = root / "captions.json"
            captions_path.write_text(json.dumps(captions), encoding="utf-8")

            pairs = module.build_pairs(
                captions_path,
                image_dir,
                limit=2,
                seed=0,
                caption_policy="all",
            )

            self.assertEqual(len(pairs), 5)
            self.assertEqual(sorted({row["image_id"] for row in pairs}), [1, 2])
            by_image = {image_id: [row for row in pairs if row["image_id"] == image_id] for image_id in [1, 2]}
            self.assertEqual([row["caption_index"] for row in by_image[1]], [0, 1])
            self.assertEqual([row["caption"] for row in by_image[2]], ["b first", "b second", "b third"])


if __name__ == "__main__":
    unittest.main()
