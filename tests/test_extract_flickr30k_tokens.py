from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_flickr30k_tokens.py"


def load_flickr_module():
    spec = importlib.util.spec_from_file_location("extract_flickr30k_tokens_for_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFlickr30kTokenExtractionPairs(unittest.TestCase):
    def test_build_pairs_reads_csv_from_zip_and_keeps_all_captions(self) -> None:
        module = load_flickr_module()
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "flickr.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("captions.txt", "image,caption\n1.jpg,first one\n1.jpg,second one\n2.jpg,first two\n")
                zf.writestr("Images/1.jpg", b"fake")
                zf.writestr("Images/2.jpg", b"fake")

            pairs = module.build_pairs(
                zip_path=zip_path,
                captions_member="captions.txt",
                image_prefix="Images/",
                limit=1,
                seed=0,
                caption_policy="all",
            )

            self.assertEqual(len(pairs), 2)
            self.assertEqual({row["file_name"] for row in pairs}, {"1.jpg"})
            self.assertEqual([row["caption_index"] for row in pairs], [0, 1])
            self.assertEqual([row["caption"] for row in pairs], ["first one", "second one"])
            self.assertEqual([row["image_id"] for row in pairs], [1, 1])
            self.assertEqual([row["zip_member"] for row in pairs], ["Images/1.jpg", "Images/1.jpg"])


if __name__ == "__main__":
    unittest.main()
