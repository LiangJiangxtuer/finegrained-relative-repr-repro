from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pal_repro.segmentation import (  # noqa: E402
    ADE20KSegmentationDataset,
    PASCAL_CONTEXT_59_CLASS_NAMES,
    PascalContextSegmentationDataset,
    VOC_CLASS_NAMES,
    build_segmentation_prompt_groups,
    calibrate_dense_logits,
    clean_ade20k_object_aliases,
    foreground_miou_from_intersections_unions,
    parse_ade20k_object_aliases,
    parse_ade20k_object_info,
    parse_pascal_context_labels,
    patch_logits_to_label_mask,
    select_pascal_context_labels,
    transform_mask_like_image_processor,
    update_intersections_unions,
)
from scripts.evaluate_segmentation import build_parser, load_dataset, manual_class_bias  # noqa: E402


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

    def test_patch_logits_to_label_mask_maps_non_contiguous_label_ids(self) -> None:
        logits = torch.tensor(
            [
                [
                    [[3.0, 1.0], [1.0, 0.0]],
                    [[0.0, 2.0], [4.0, 5.0]],
                ]
            ]
        )

        pred = patch_logits_to_label_mask(logits, output_size=(2, 2), label_ids=[2, 9])

        self.assertEqual(pred.tolist(), [[[2, 9], [9, 9]]])

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

    def test_select_pascal_context_common59_maps_raw_label_ids(self) -> None:
        labels = [(idx + 1, name) for idx, name in enumerate(PASCAL_CONTEXT_59_CLASS_NAMES)]

        selected = select_pascal_context_labels(labels, protocol="common59")

        self.assertEqual(len(selected), 59)
        self.assertEqual(selected[0], (1, "aeroplane"))
        self.assertEqual(selected[-1], (59, "wood"))

    def test_evaluate_segmentation_parser_exposes_target_frame_and_context_protocol(self) -> None:
        args = build_parser().parse_args(
            [
                "--dataset", "Context",
                "--checkpoint", "checkpoint.pt",
                "--output", "out.json",
                "--target-frame", "processor",
                "--context-protocol", "common59",
            ]
        )

        self.assertEqual(args.target_frame, "processor")
        self.assertEqual(args.context_protocol, "common59")

    def test_load_context_dataset_common59_uses_selected_raw_ids_and_prompt_names(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trainval").mkdir()
            rows = [f"{idx + 10}: {name}" for idx, name in enumerate(PASCAL_CONTEXT_59_CLASS_NAMES)]
            (root / "labels.txt").write_text("\n".join(rows), encoding="utf-8")

            _dataset, class_names, class_ids, _ignore, _split = load_dataset(
                "Context",
                root,
                context_protocol="common59",
            )

        self.assertEqual(len(class_ids), 59)
        self.assertEqual(class_ids[0], 10)
        self.assertEqual(class_names[PASCAL_CONTEXT_59_CLASS_NAMES.index("tvmonitor")], "tv monitor")

    def test_parse_ade20k_object_info_uses_first_name_alias(self) -> None:
        rows = [
            "Idx\tRatio\tTrain\tVal\tName",
            "1\t0.1\t10\t1\twall",
            "2\t0.1\t10\t1\tbuilding, edifice",
        ]

        labels = parse_ade20k_object_info(rows)

        self.assertEqual(labels, [(1, "wall"), (2, "building")])

    def test_parse_ade20k_object_aliases_keeps_all_prompt_synonyms(self) -> None:
        rows = [
            "Idx\tRatio\tTrain\tVal\tName",
            "2\t0.1\t10\t1\tbuilding, edifice",
            "3\t0.1\t10\t1\trug, carpet, carpeting",
        ]

        aliases = parse_ade20k_object_aliases(rows)

        self.assertEqual(aliases[0], (2, ["building", "edifice"]))
        self.assertEqual(aliases[1], (3, ["rug", "carpet", "carpeting"]))

    def test_build_segmentation_prompt_groups_combines_aliases_and_templates(self) -> None:
        groups = build_segmentation_prompt_groups(
            [["building", "edifice"], ["rug"]],
            templates=["a photo of {class_name}", "a clean photo of {class_name}"],
        )

        self.assertEqual(
            groups,
            [
                [
                    "a photo of building",
                    "a clean photo of building",
                    "a photo of edifice",
                    "a clean photo of edifice",
                ],
                ["a photo of rug", "a clean photo of rug"],
            ],
        )

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

    def test_transform_mask_like_image_processor_matches_resize_center_crop_geometry(self) -> None:
        from PIL import Image

        mask = Image.fromarray(
            np.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [10, 11, 12, 13, 14, 15],
                    [20, 21, 22, 23, 24, 25],
                    [30, 31, 32, 33, 34, 35],
                ],
                dtype=np.uint8,
            )
        )
        processor = SimpleNamespace(
            do_resize=True,
            size={"shortest_edge": 4},
            do_center_crop=True,
            crop_size={"height": 2, "width": 2},
        )

        transformed = transform_mask_like_image_processor(mask, processor)

        self.assertEqual(np.asarray(transformed).tolist(), [[12, 13], [22, 23]])

    def test_evaluate_segmentation_parser_accepts_layer_and_prompt_ensemble_flags(self) -> None:
        args = build_parser().parse_args(
            [
                "--dataset", "ADE20K",
                "--checkpoint", "checkpoint.pt",
                "--output", "out.json",
                "--vision-layer", "-2",
                "--text-layer", "-6",
                "--vision-layer-ensemble", "-1",
                "--vision-layer-ensemble", "-2",
                "--text-layer-ensemble", "-2",
                "--text-layer-ensemble", "-4",
                "--image-size", "448",
                "--ignore-zero",
                "--alias-policy", "all",
                "--class-prior-source", "ade20k-ratio",
                "--class-prior-alpha", "0.25",
                "--class-bias", "wall,sky=0.02",
                "--class-bias", "screen door=-0.03",
                "--logit-calibration", "image-class-zscore",
                "--prompt-template", "a photo of {class_name}",
                "--prompt-template", "a clean photo of {class_name}",
            ]
        )

        self.assertEqual(args.vision_layer, -2)
        self.assertEqual(args.text_layer, -6)
        self.assertEqual(args.vision_layer_ensemble, [-1, -2])
        self.assertEqual(args.text_layer_ensemble, [-2, -4])
        self.assertEqual(args.image_size, 448)
        self.assertTrue(args.ignore_zero)
        self.assertEqual(args.alias_policy, "all")
        self.assertEqual(args.class_prior_source, "ade20k-ratio")
        self.assertAlmostEqual(args.class_prior_alpha, 0.25)
        self.assertEqual(args.class_bias, ["wall,sky=0.02", "screen door=-0.03"])
        self.assertEqual(args.logit_calibration, "image-class-zscore")
        self.assertEqual(args.prompt_template, ["a photo of {class_name}", "a clean photo of {class_name}"])

    def test_manual_class_bias_supports_explicit_groups_and_repeated_specs(self) -> None:
        bias = manual_class_bias(
            ["wall", "building", "sky", "screen door"],
            ["wall,sky=0.02", "screen door=-0.03", "wall=0.01"],
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(bias)
        assert bias is not None
        self.assertEqual(tuple(bias.shape), (1, 4, 1, 1))
        self.assertTrue(torch.allclose(bias.flatten(), torch.tensor([0.03, 0.0, 0.02, -0.03])))

    def test_calibrate_dense_logits_centers_and_zscores_per_image_class(self) -> None:
        logits = torch.tensor(
            [
                [
                    [[1.0, 3.0], [5.0, 7.0]],
                    [[2.0, 2.0], [2.0, 2.0]],
                ]
            ]
        )

        centered = calibrate_dense_logits(logits, mode="image-class-center")
        zscored = calibrate_dense_logits(logits, mode="image-class-zscore")

        self.assertTrue(torch.allclose(centered.mean(dim=(2, 3)), torch.zeros(1, 2)))
        self.assertTrue(torch.allclose(zscored[0, 0].std(unbiased=False), torch.tensor(1.0)))
        self.assertTrue(torch.allclose(zscored[0, 1], torch.zeros(2, 2)))

    def test_clean_ade20k_object_aliases_removes_broad_wordnet_synonyms(self) -> None:
        aliases = clean_ade20k_object_aliases(
            [
                (7, ["road", "route"]),
                (13, ["person", "individual", "someone", "somebody", "mortal", "soul"]),
                (19, ["curtain", "drape", "drapery", "mantle", "pall"]),
                (21, ["car", "auto", "automobile", "machine", "motorcar"]),
                (66, ["toilet", "can", "commode", "crapper", "pot", "potty", "stool", "throne"]),
                (88, ["television receiver", "television", "tv", "idiot box"]),
                (128, ["bicycle", "bike", "wheel", "cycle"]),
                (139, ["ashcan", "trash can", "garbage can", "ash bin", "dustbin", "trash bin"]),
            ]
        )

        self.assertEqual(aliases[0], (7, ["road"]))
        self.assertEqual(aliases[1], (13, ["person"]))
        self.assertEqual(aliases[2], (19, ["curtain", "drape", "drapery"]))
        self.assertEqual(aliases[3], (21, ["car", "automobile", "motorcar"]))
        self.assertEqual(aliases[4], (66, ["toilet", "commode"]))
        self.assertEqual(aliases[5], (88, ["television", "tv"]))
        self.assertEqual(aliases[6], (128, ["bicycle", "bike"]))
        self.assertEqual(aliases[7], (139, ["ashcan", "trash can", "garbage can", "dustbin", "trash bin"]))


if __name__ == "__main__":
    unittest.main()
