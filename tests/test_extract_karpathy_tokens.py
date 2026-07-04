from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_extract_module():
    spec = importlib.util.spec_from_file_location(
        "extract_karpathy_retrieval_tokens",
        ROOT / "scripts" / "extract_karpathy_retrieval_tokens.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestKarpathyRetrievalPairs(unittest.TestCase):
    def test_build_coco_pairs_uses_requested_split_and_all_captions(self) -> None:
        module = load_extract_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "val2014"
            image_dir.mkdir(parents=True)
            (image_dir / "COCO_val2014_000000000042.jpg").write_bytes(b"fake")
            payload = {
                "images": [
                    {
                        "split": "test",
                        "filepath": "val2014",
                        "filename": "COCO_val2014_000000000042.jpg",
                        "imgid": 42,
                        "sentences": [
                            {"raw": "a caption", "sentid": 1001},
                            {"raw": "another caption", "sentid": 1002},
                        ],
                    },
                    {
                        "split": "train",
                        "filepath": "train2014",
                        "filename": "COCO_train2014_000000000007.jpg",
                        "imgid": 7,
                        "sentences": [{"raw": "not selected", "sentid": 1}],
                    },
                ]
            }
            path = root / "dataset_coco.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            rows = module.build_karpathy_pairs(
                karpathy_json=path,
                dataset="coco",
                split="test",
                coco_root=root,
                flickr_zip=None,
                caption_policy="all",
                limit_images=None,
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["image_id"] for row in rows}, {42})
        self.assertEqual([row["caption_index"] for row in rows], [0, 1])
        self.assertTrue(rows[0]["image_path"].endswith("val2014/COCO_val2014_000000000042.jpg"))
        self.assertEqual(rows[0]["karpathy_split"], "test")
        self.assertEqual(rows[0]["caption_policy"], "all")

    def test_build_flickr_pairs_resolves_zip_member_and_limit(self) -> None:
        module = load_extract_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import zipfile

            zip_path = root / "flickr.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("Images/1.jpg", b"fake")
                zf.writestr("Images/2.jpg", b"fake")
            payload = {
                "images": [
                    {
                        "split": "test",
                        "filename": "1.jpg",
                        "imgid": 1,
                        "sentences": [{"raw": "first", "sentid": 11}, {"raw": "second", "sentid": 12}],
                    },
                    {
                        "split": "test",
                        "filename": "2.jpg",
                        "imgid": 2,
                        "sentences": [{"raw": "third", "sentid": 21}],
                    },
                ]
            }
            path = root / "dataset_flickr30k.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            rows = module.build_karpathy_pairs(
                karpathy_json=path,
                dataset="flickr30k",
                split="test",
                coco_root=None,
                flickr_zip=zip_path,
                caption_policy="first",
                limit_images=1,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_id"], 1)
        self.assertEqual(rows[0]["zip_member"], "Images/1.jpg")
        self.assertEqual(rows[0]["caption"], "first")


if __name__ == "__main__":
    unittest.main()
