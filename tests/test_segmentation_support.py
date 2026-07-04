from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.segmentation import (  # noqa: E402
    ADE20KSegmentationDataset,
    PascalContextSegmentationDataset,
    VOC_CLASS_NAMES,
    foreground_miou_from_intersections_unions,
    parse_ade20k_object_info,
    parse_pascal_context_labels,
    patch_logits_to_label_mask,
    update_intersections_unions,
)


class TestSegmentationSupport(unittest.TestCase):
    def test_voc_class_names_match_20_foreground_ids(self) -> None:
        self.assertEqual(len(VOC_CLASS_NAMES), 20)
        self.assertEqual(VOC_CLASS_NAMES[:3], ["aeroplane", "bicycle", "bird"])
        self.assertEqual(VOC_CLASS_NAMES[-1], "tv monitor")

    def test_patch_logits_to_label_mask_upsamples_and_offsets_foreground_labels(self) -> None:
        logits = torch.tensor(
            [
                [
                    [[3.0, 1.0], [1.0, 0.0]],
                    [[0.0, 2.0], [4.0, 5.0]],
                ]
            ]
        )  # (B=1, C=2, H=2, W=2)

        pred = patch_logits_to_label_mask(logits, output_size=(4, 4), label_offset=1)

        self.assertEqual(tuple(pred.shape), (1, 4, 4))
        self.assertEqual(int(pred.min()), 1)
        self.assertEqual(int(pred.max()), 2)

    def test_intersection_union_accumulator_ignores_background_and_255(self) -> None:
        pred = torch.tensor([[1, 2, 2], [1, 1, 2]])
        target = torch.tensor([[1, 2, 0], [255, 2, 2]])
        intersections = torch.zeros(2, dtype=torch.float64)
        unions = torch.zeros(2, dtype=torch.float64)

        update_intersections_unions(
            intersections,
            unions,
            pred,
            target,
            class_ids=[1, 2],
            ignore_index=255,
        )
        miou = foreground_miou_from_intersections_unions(intersections, unions)

        self.assertTrue(torch.equal(intersections, torch.tensor([1.0, 2.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(unions, torch.tensor([2.0, 4.0], dtype=torch.float64)))
        self.assertAlmostEqual(miou, 50.0)

    def test_parse_pascal_context_labels_keeps_numeric_ids(self) -> None:
        labels = parse_pascal_context_labels(["1: accordion", "23: bicycle", "459: wood"])

        self.assertEqual(labels[0], (1, "accordion"))
        self.assertEqual(labels[1], (23, "bicycle"))
        self.assertEqual(labels[-1], (459, "wood"))

    def test_parse_ade20k_object_info_uses_first_name_alias(self) -> None:
        rows = [
            "Idx\tRatio\tTrain\tVal\tName",
            "1\t0.1\t10\t1\twall",
            "2\t0.1\t10\t1\tbuilding, edifice",
        ]

        labels = parse_ade20k_object_info(rows)

        self.assertEqual(labels, [(1, "wall"), (2, "building")])

    def test_segmentation_dataset_roots_expose_class_ids_from_metadata(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            context_root = Path(tmp) / "pascal_context" / "raw"
            context_root.mkdir(parents=True)
            (context_root / "labels.txt").write_text("1: accordion\n2: aeroplane\n", encoding="utf-8")
            context = PascalContextSegmentationDataset(context_root)

            ade_root = Path(tmp) / "ADEChallengeData2016"
            ade_root.mkdir(parents=True)
            (ade_root / "objectInfo150.txt").write_text(
                "Idx\tRatio\tTrain\tVal\tName\n1\t0\t0\t0\twall\n2\t0\t0\t0\tbuilding, edifice\n",
                encoding="utf-8",
            )
            ade = ADE20KSegmentationDataset(ade_root)

        self.assertEqual(context.class_ids, [1, 2])
        self.assertEqual(context.class_names, ["accordion", "aeroplane"])
        self.assertEqual(ade.class_ids, [1, 2])
        self.assertEqual(ade.class_names, ["wall", "building"])


if __name__ == "__main__":
    unittest.main()
