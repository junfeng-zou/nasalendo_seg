"""Controls that matter for the distance-state experiment, using synthetic image fixtures."""
import math
import csv
import tempfile
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from distance_state_classifier_endodac.src.appearance_consistency import (
    appearance_regions, augment_appearance, jensen_shannon_logits, paired_objective,
)
from distance_state_classifier_endodac.src.mask_utils import mask_cache_path, write_mask
from distance_state_classifier_endodac.src.config import load_config
from distance_state_classifier_endodac.scripts.train import make_dataset
from distance_state_classifier_endodac.scripts.train_appearance_consistency import (
    DEFAULT_CONFIG, pair_dataset, run_epoch,
)


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((120, 160, 3), np.uint8)
        cv2.circle(self.image, (80, 60), 54, (165, 105, 45), -1)
        self.mask = np.zeros((120, 160), np.uint8)
        self.mask[40:85, 72:95] = 255
        self.image[self.mask > 0] = (95, 70, 160)
        self.cfg = dict(guard_px=2, identity_prob=0, bg_prob=1, fg_prob=1, luminance_prob=0)

    def test_geometry_boundary_border_and_source_are_preserved(self):
        image_before, mask_before = self.image.copy(), self.mask.copy()
        regions = appearance_regions(self.image, self.mask)
        changed, _ = augment_appearance(self.image, self.mask, self.cfg, np.random.default_rng(42))
        protected = ~(regions["fg"] | regions["bg"])
        self.assertEqual(changed.shape, self.image.shape)
        self.assertEqual(changed.dtype, self.image.dtype)
        np.testing.assert_array_equal(changed[protected], self.image[protected])
        np.testing.assert_array_equal(image_before, self.image)
        np.testing.assert_array_equal(mask_before, self.mask)
        self.assertTrue(np.any(changed[regions["bg"]] != self.image[regions["bg"]]))
        self.assertTrue(np.any(changed[regions["fg"]] != self.image[regions["fg"]]))

    def test_background_only_keeps_every_instrument_pixel(self):
        cfg = dict(self.cfg, fg_prob=0)
        changed, _ = augment_appearance(self.image, self.mask, cfg, np.random.default_rng(8))
        np.testing.assert_array_equal(changed[self.mask > 0], self.image[self.mask > 0])

    def test_foreground_only_keeps_background(self):
        cfg = dict(self.cfg, bg_prob=0)
        changed, _ = augment_appearance(self.image, self.mask, cfg, np.random.default_rng(8))
        np.testing.assert_array_equal(changed[self.mask == 0], self.image[self.mask == 0])

    def test_missing_region_is_not_invented(self):
        empty = np.zeros_like(self.mask)
        changed, info = augment_appearance(self.image, empty, self.cfg, np.random.default_rng(42))
        np.testing.assert_array_equal(changed, self.image)
        self.assertFalse(info["mask_nonempty"])
        thin = empty.copy()
        thin[40:85, 80] = 255
        regions = appearance_regions(self.image, thin)
        self.assertFalse(regions["fg"].any())
        self.assertTrue(regions["bg"].any())
        with self.assertRaises(ValueError):
            appearance_regions(self.image, thin[:10])

    def test_seed_reproducibility(self):
        first, _ = augment_appearance(self.image, self.mask, self.cfg, np.random.default_rng(71))
        second, _ = augment_appearance(self.image, self.mask, self.cfg, np.random.default_rng(71))
        np.testing.assert_array_equal(first, second)

    def test_clean_view_is_exactly_baseline_preprocessing(self):
        config = load_config(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, label in enumerate(config["data"]["classes"]):
                path = root / f"sample_{index}.png"
                image = np.roll(self.image, index * 3, axis=1)
                mask = np.roll(self.mask, index * 3, axis=1)
                self.assertTrue(cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
                write_mask(mask_cache_path(root / "masks", path), mask)
                rows.append({"image_path": str(path), "label": label})
            csv_path = root / "train.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "label"])
                writer.writeheader()
                writer.writerows(rows)
            config["data"]["train_csv"] = str(csv_path)
            config["appearance_consistency"]["mask_cache"] = str(root / "masks")
            paired = pair_dataset(config)
            baseline = make_dataset(config, "train", False)
            self.assertEqual(len(paired), 3)
            for index in range(len(paired)):
                views, target, _ = paired[index]
                clean, original_target, _ = baseline[index]
                self.assertTrue(torch.equal(views[0], clean))
                self.assertEqual(target.item(), original_target.item())
            with self.assertRaisesRegex(ValueError, "legacy"):
                config["augmentation"]["enabled"] = True
                pair_dataset(config)



class LossTests(unittest.TestCase):
    def test_zero_for_identical_and_symmetric_for_different_logits(self):
        a = torch.tensor([[2., 0., -1.], [-1., 0., 2.]])
        b = -a
        self.assertAlmostEqual(jensen_shannon_logits(a, a).item(), 0, places=7)
        self.assertAlmostEqual(jensen_shannon_logits(a, b).item(), jensen_shannon_logits(b, a).item(), places=7)
        js = jensen_shannon_logits(torch.tensor([[10000., -10000., 0.]]), torch.tensor([[-10000., 10000., 0.]]))
        self.assertTrue(torch.isfinite(js))
        self.assertAlmostEqual(js.item(), math.log(2), places=6)

    def test_both_views_receive_gradients_and_ce_keeps_class_weights(self):
        a = torch.tensor([[1., 0., -1.], [0., 1., 2.]], requires_grad=True)
        b = torch.tensor([[0., 1., -1.], [1., 0., 2.]], requires_grad=True)
        labels = torch.tensor([0, 2])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([.45, 1.15, 1.35]))
        total, ce, js = paired_objective(a, b, labels, criterion, 1.)
        expected = .5 * (criterion(a, labels) + criterion(b, labels))
        torch.testing.assert_close(ce, expected)
        torch.testing.assert_close(total, expected + js)
        js.backward(retain_graph=True)
        self.assertGreater(a.grad.abs().sum().item(), 0)
        self.assertGreater(b.grad.abs().sum().item(), 0)
        a.grad = b.grad = None
        total.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())

    def test_paired_training_updates_model_and_clean_eval_does_not(self):
        class TinyDataset(Dataset):
            classes = ["TooFar", "Good", "TooClose"]
            def __init__(self, paired):
                self.paired = paired
            def __len__(self):
                return 6
            def __getitem__(self, index):
                clean = torch.full((3, 4, 4), float(index % 3))
                views = torch.stack([clean, clean + .1]) if self.paired else clean
                return views, torch.tensor(index % 3), {}
        torch.manual_seed(3)
        model = nn.Sequential(nn.Flatten(), nn.Linear(48, 3))
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        criterion = nn.CrossEntropyLoss()
        before = model[1].weight.detach().clone()
        metrics = run_epoch(model, DataLoader(TinyDataset(True), batch_size=2), criterion, torch.device("cpu"),
                            optimizer=optimizer, scaler=torch.cuda.amp.GradScaler(enabled=False))
        self.assertEqual(metrics["n"], 6)
        self.assertFalse(torch.equal(before, model[1].weight))
        before = model[1].weight.detach().clone()
        evaluation = run_epoch(model, DataLoader(TinyDataset(False), batch_size=2), criterion, torch.device("cpu"))
        self.assertEqual(evaluation["js"], 0)
        self.assertTrue(torch.equal(before, model[1].weight))


if __name__ == "__main__":
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    unittest.main()
