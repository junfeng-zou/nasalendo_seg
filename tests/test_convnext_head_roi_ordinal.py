"""Check the scale controls, ordered probabilities, and abstention accounting."""
import math
import tempfile
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from distance_state_classifier.src.head_roi import crop_with_padding, estimate_fov, locate_roi
from distance_state_classifier.src.convnext_roi_ordinal import ConvNeXtROIOrdinal, ordered_log_probabilities, ordinal_loss
from distance_state_classifier.scripts.train_head_roi_ordinal import summarize_predictions, HeadRoiDataset
from distance_state_classifier.src.config import load_config
from distance_state_classifier.src.transforms import resize_and_normalize


class RoiTests(unittest.TestCase):
    def test_cached_roi_matches_raw_image_inference_preprocessing(self):
        config = load_config("distance_state_classifier/configs/convnext_head_roi_ordinal.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            rng = np.random.default_rng(42)
            boxes = [[10, 15, 90, 95], [-10, 5, 80, 95], [80, 60, 180, 160]]
            for index, (label, box) in enumerate(zip(config["data"]["classes"], boxes)):
                original = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
                image_path, roi_path = root / f"image_{index}.png", root / f"roi_{index}.png"
                self.assertTrue(cv2.imwrite(str(image_path), original))
                self.assertTrue(cv2.imwrite(str(roi_path), crop_with_padding(original, box)))
                records.append(dict(split="val", label=label, image_path=str(image_path),
                                    roi_path=str(roi_path), geometry=dict(valid=True, box=box)))
            dataset = HeadRoiDataset(records, config, "val")
            self.assertEqual(len(dataset), 3)
            for index, row in enumerate(records):
                original = cv2.imread(row["image_path"])
                roi = crop_with_padding(original, row["geometry"]["box"])
                raw_input = resize_and_normalize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), config["image"]["input_size"])
                cached_input, _, valid, _ = dataset[index]
                self.assertTrue(valid)
                np.testing.assert_array_equal(cached_input.numpy(), raw_input)

    def test_padding_preserves_canvas_and_pixel_scale(self):
        image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        crop = crop_with_padding(image, [-2, -1, 8, 7])
        self.assertEqual(crop.shape, (8, 10, 3))
        np.testing.assert_array_equal(crop[1:6, 2:9], image)
        self.assertTrue((crop[0] == 0).all())
        self.assertTrue((crop[:, :2] == 0).all())
        self.assertEqual(crop_with_padding(image, [20, 20, 30, 30]).sum(), 0)
        with self.assertRaises(ValueError):
            crop_with_padding(image, [3, 3, 2, 4])

    def test_optical_fov_estimation_on_clipped_circle(self):
        image = np.zeros((180, 220, 3), np.uint8)
        cv2.circle(image, (110, 90), 98, (80, 100, 130), -1)
        fov = estimate_fov(image)
        np.testing.assert_allclose(fov["center"], [110, 90], atol=2)
        self.assertAlmostEqual(fov["diameter"], 196, delta=3)
        with self.assertRaises(ValueError):
            estimate_fov(np.zeros_like(image))

    def test_fixed_fov_window_does_not_resize_with_instrument(self):
        cfg = load_config("distance_state_classifier/configs/convnext_head_roi_ordinal.yaml")["roi"]
        fov = {"shape": [240, 320], "center": [160, 120], "diameter": 230}
        sizes = []
        for width in (20, 40):
            mask = np.zeros((240, 320), np.uint8)
            mask[70:238, 160 - width // 2:160 + width // 2] = 255
            roi = locate_roi(mask, fov, cfg)
            self.assertIn("box", roi)
            sizes.append((roi["box"][2] - roi["box"][0], roi["box"][3] - roi["box"][1]))
        self.assertEqual(sizes, [(92, 92), (92, 92)])

    def test_missing_mask_does_not_create_a_head(self):
        cfg = load_config("distance_state_classifier/configs/convnext_head_roi_ordinal.yaml")["roi"]
        roi = locate_roi(np.zeros((240, 320), np.uint8), {"shape": [240, 320], "center": [160, 120], "diameter": 230}, cfg)
        self.assertFalse(roi["valid"])
        self.assertNotIn("box", roi)


class OrdinalTests(unittest.TestCase):
    def test_initial_thresholds_allow_all_three_decisions(self):
        torch.set_num_threads(2)
        model = ConvNeXtROIOrdinal()
        gap = torch.nn.functional.softplus(model.raw_gap.detach()) + 1e-4
        score = torch.tensor([-5., 0., 5.])
        logits = score[:, None] - torch.stack([-gap/2, gap/2])
        self.assertEqual(ordered_log_probabilities(logits).argmax(-1).tolist(), [0, 1, 2])

    def test_probabilities_are_normalized_and_ordered(self):
        score = torch.tensor([-1000., -5., 0., 5., 1000.])
        logits = score[:, None] - torch.tensor([-1., 1.])
        log_probs = ordered_log_probabilities(logits)
        self.assertTrue(torch.isfinite(log_probs).all())
        probs = log_probs.exp()
        torch.testing.assert_close(probs.sum(-1), torch.ones(5))
        self.assertTrue((logits.sigmoid()[:, 0] >= logits.sigmoid()[:, 1]).all())
        self.assertEqual(probs.argmax(-1).tolist(), [0, 0, 1, 2, 2])

    def test_middle_probability_matches_cumulative_difference(self):
        z = torch.tensor([[.5, -.3], [4., 1.], [-1., -3.]])
        p = ordered_log_probabilities(z).exp()
        q = z.sigmoid()
        torch.testing.assert_close(p, torch.stack([1-q[:, 0], q[:, 0]-q[:, 1], q[:, 1]], -1))

    def test_ordinal_loss_penalizes_opposite_state_more(self):
        far = torch.tensor([[-4., -6.]])
        losses = [ordinal_loss(far, torch.tensor([label])).item() for label in range(3)]
        self.assertLess(losses[0], losses[1])
        self.assertLess(losses[1], losses[2])

    def test_weighted_ordinal_objective_and_gradients(self):
        score = torch.tensor([-.3, .8], requires_grad=True)
        gap = torch.tensor(1., requires_grad=True)
        logits = score[:, None] - torch.stack([-gap/2, gap/2])
        labels = torch.tensor([0, 2])
        weights = torch.tensor([2., 1., 3.])
        loss = ordinal_loss(logits, labels, weights)
        expected = (2 * ordinal_loss(logits[:1], labels[:1]) + 3 * ordinal_loss(logits[1:], labels[1:])) / 5
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(score.grad).all())
        self.assertTrue(torch.isfinite(gap.grad))
        self.assertTrue((score.grad.abs() > 0).all())

    def test_unavailable_roi_remains_in_evaluation_denominator(self):
        metrics = summarize_predictions([0, 1, 2, 2], [0, 1, 2, 3], ["TooFar", "Good", "TooClose"])
        self.assertEqual(metrics["accuracy"], .75)
        self.assertEqual(metrics["coverage"], .75)
        self.assertEqual(metrics["per_class"][2]["recall"], .5)
        self.assertEqual(metrics["per_class"][2]["support"], 2)
        self.assertEqual(metrics["invalid_n"], 1)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
