"""Validate the experimental controls and paired diagnostic metrics."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_distance_shortcuts import (
    appearance_interventions,
    intervention_regions,
    probability_distance,
    select_rows,
    summarize,
)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(self.image, (80, 60), 53, (45, 105, 165), -1)
        self.mask = np.zeros(self.image.shape[:2], np.uint8)
        self.mask[40:85, 72:90] = 255
        self.image[self.mask > 0] = (160, 70, 95)
        self.regions = intervention_regions(self.image, self.mask, 3)

    def test_regional_edits_preserve_protected_pixels_and_geometry(self):
        before = self.image.copy()
        variants = appearance_interventions(self.image, self.regions)
        for name, image in variants.items():
            self.assertEqual(image.shape, self.image.shape)
            if name.startswith("bg_"):
                np.testing.assert_array_equal(image[~self.regions["bg"]], before[~self.regions["bg"]])
                np.testing.assert_array_equal(image[self.mask > 0], before[self.mask > 0])
            if name.startswith("fg_"):
                np.testing.assert_array_equal(image[~self.regions["fg"]], before[~self.regions["fg"]])
            np.testing.assert_array_equal(image[~self.regions["fov"]], before[~self.regions["fov"]])
        np.testing.assert_array_equal(self.image, before)
        np.testing.assert_array_equal(variants["original"], before)
        self.assertTrue(np.any(variants["bg_chroma_plus"] != before))
        self.assertTrue(np.any(variants["fg_chroma_plus"] != before))

    def test_empty_mask_rejected_and_thin_foreground_not_fabricated(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            intervention_regions(self.image, np.zeros_like(self.mask))
        thin = np.zeros_like(self.mask)
        thin[40:85, 80] = 255
        regions = intervention_regions(self.image, thin, 3)
        self.assertFalse(regions["fg"].any())
        variants = appearance_interventions(self.image, regions)
        np.testing.assert_array_equal(variants["fg_chroma_plus"], self.image)

    def test_probability_distance(self):
        self.assertAlmostEqual(probability_distance([0.8, 0.1, 0.1], [0.2, 0.7, 0.1]), 0.6)
        self.assertEqual(probability_distance([1, 0, 0], [0, 0, 1]), 1)

    def test_paired_metrics_distinguish_flips_from_errors(self):
        classes = ["TooFar", "Good", "TooClose"]
        rows = []
        # First pair changes a correct prediction into an error; second fixes an error.
        for truth, original, changed in [("TooFar", "TooFar", "Good"), ("TooClose", "Good", "TooClose")]:
            for variant, prediction in [("original", original), ("bg_desaturate", changed)]:
                rows.append({"label": truth, "prediction": prediction, "original_prediction": original,
                             "variant": variant, "intervention_valid": True, "flipped": prediction != original,
                             "probability_tv": 0.3 if prediction != original else 0,
                             "changed_pixel_fraction": 0.5, "mean_pixel_delta_255": 20,
                             "opposite_extreme_flip": False})
        result = summarize(rows, classes)["variants"]["bg_desaturate"]
        self.assertEqual(result["flip_rate"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["accuracy_delta"], 0)
        self.assertEqual(result["correct_to_wrong_rate"], 1)
        self.assertEqual(result["original_correct_n"], 1)
        for row in rows:
            row["label"] = ""
        self.assertIsNone(summarize(rows, classes)["variants"]["bg_desaturate"]["accuracy"])

    def test_invalid_region_excluded_from_summary(self):
        rows = [{"variant": "fg_desaturate", "intervention_valid": False}]
        result = summarize(rows, ["TooFar", "Good", "TooClose"])["variants"]["fg_desaturate"]
        self.assertEqual(result, {"n": 0, "unavailable_n": 1})

    def test_small_sample_is_deterministic_and_group_balanced(self):
        rows = [{"video_id": "v", "label": label, "image_path": str(i)} for label in ["A", "B", "C"] for i in range(10)]
        sampled = select_rows(rows, 6, 42)
        self.assertEqual(sampled, select_rows(rows, 6, 42))
        self.assertEqual([sum(r["label"] == c for r in sampled) for c in ["A", "B", "C"]], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()
