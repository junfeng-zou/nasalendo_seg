"""大面积高反光实例过滤的轻量单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


INFERENCE_DIR = Path(__file__).resolve().parents[1] / "inference"
sys.path.insert(0, str(INFERENCE_DIR))

import realtime_improved_tip as tip_runtime  # noqa: E402


class _FakeMasks:
    def __init__(self, masks):
        self.data = masks

    def __len__(self):
        return len(self.data)


class _FakeBoxes:
    def __init__(self, confidences, boxes):
        self.conf = np.asarray(confidences, dtype=np.float32)
        self.xyxy = np.asarray(boxes, dtype=np.float32)

    def __len__(self):
        return len(self.conf)


class _FakeResult:
    def __init__(self, masks, confidences=None, boxes=None):
        self.masks = _FakeMasks(masks)
        self.boxes = (
            None
            if confidences is None
            else _FakeBoxes(confidences, boxes)
        )


def _rect_mask(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


class SpecularInstanceFilterTests(unittest.TestCase):
    def test_default_photometric_boundary_is_v250_s8_and_inclusive(self):
        hsv = np.array(
            [[[0, 8, 250], [0, 8, 249], [0, 9, 250]]],
            dtype=np.uint8,
        )
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mask = np.ones((1, 3), dtype=np.uint8)

        metrics = tip_runtime._specular_instance_metrics(frame, mask)

        self.assertEqual(metrics["specular_area"], 1)

    def test_large_contiguous_white_instance_is_rejected(self):
        shape = (100, 100)
        mask = _rect_mask(shape, 15, 15, 85, 85)
        frame = np.zeros((*shape, 3), dtype=np.uint8)
        frame[mask > 0] = (255, 255, 255)
        result = _FakeResult([mask], [0.95], [[15, 15, 84, 84]])

        detection, diagnostics = tip_runtime._select_detection(
            result,
            frame.shape,
            min_area_ratio=0.001,
            morph_kernel=1,
            frame=frame,
            specular_filter_enabled=True,
            return_diagnostics=True,
        )

        self.assertIsNone(detection)
        self.assertEqual(diagnostics["candidate_count"], 1)
        self.assertEqual(diagnostics["specular_rejected_count"], 1)

    def test_small_metallic_highlight_does_not_remove_instrument(self):
        shape = (100, 100)
        mask = _rect_mask(shape, 42, 10, 58, 90)
        frame = np.full((*shape, 3), (60, 60, 150), dtype=np.uint8)
        frame[35:40, 45:50] = (255, 255, 255)
        result = _FakeResult([mask], [0.92], [[42, 10, 57, 89]])

        detection, diagnostics = tip_runtime._select_detection(
            result,
            frame.shape,
            min_area_ratio=0.001,
            morph_kernel=1,
            frame=frame,
            specular_filter_enabled=True,
            return_diagnostics=True,
        )

        self.assertIsNotNone(detection)
        self.assertLess(detection["largest_specular_ratio"], 0.10)
        self.assertEqual(diagnostics["specular_rejected_count"], 0)

    def test_filter_can_be_disabled_for_runtime_fallback(self):
        shape = (100, 100)
        mask = _rect_mask(shape, 15, 15, 85, 85)
        frame = np.zeros((*shape, 3), dtype=np.uint8)
        frame[mask > 0] = (255, 255, 255)
        result = _FakeResult([mask], [0.95], [[15, 15, 84, 84]])

        detection = tip_runtime._select_detection(
            result,
            frame.shape,
            min_area_ratio=0.001,
            morph_kernel=1,
            frame=frame,
            specular_filter_enabled=False,
        )

        self.assertIsNotNone(detection)

    def test_moderate_specular_fraction_is_kept_by_conservative_defaults(self):
        shape = (100, 100)
        mask = _rect_mask(shape, 15, 15, 85, 85)
        frame = np.full((*shape, 3), (60, 60, 150), dtype=np.uint8)
        frame[25:60, 25:60] = (255, 255, 255)
        result = _FakeResult([mask], [0.95], [[15, 15, 84, 84]])

        detection = tip_runtime._select_detection(
            result,
            frame.shape,
            min_area_ratio=0.001,
            morph_kernel=1,
            frame=frame,
            specular_filter_enabled=True,
        )

        self.assertIsNotNone(detection)
        self.assertGreater(detection["largest_specular_ratio"], 0.20)

    def test_track_consistency_protects_even_a_highly_reflective_candidate(self):
        shape = (100, 100)
        mask = _rect_mask(shape, 15, 15, 85, 85)
        frame = np.zeros((*shape, 3), dtype=np.uint8)
        frame[mask > 0] = (255, 255, 255)
        bbox = np.array([15, 15, 84, 84], dtype=np.float64)
        result = _FakeResult([mask], [0.95], [bbox])

        detection = tip_runtime._select_detection(
            result,
            frame.shape,
            min_area_ratio=0.001,
            morph_kernel=1,
            frame=frame,
            previous_bbox=bbox,
            track_iou_threshold=0.3,
            specular_filter_enabled=True,
        )

        self.assertIsNotNone(detection)
        self.assertTrue(detection["track_consistent"])

    def test_previous_bbox_gives_consistent_candidate_a_small_bonus(self):
        shape = (100, 100)
        previous = _rect_mask(shape, 10, 20, 30, 70)
        newcomer = _rect_mask(shape, 65, 20, 85, 70)
        frame = np.full((*shape, 3), (50, 80, 140), dtype=np.uint8)
        result = _FakeResult(
            [previous, newcomer],
            [0.85, 0.95],
            [[10, 20, 29, 69], [65, 20, 84, 69]],
        )

        detection = tip_runtime._select_detection(
            result,
            frame.shape,
            min_area_ratio=0.001,
            morph_kernel=1,
            frame=frame,
            previous_bbox=np.array([10, 20, 29, 69]),
            specular_filter_enabled=True,
        )

        self.assertGreater(detection["track_iou"], 0.99)
        self.assertAlmostEqual(detection["confidence"], 0.85, places=5)

    def test_metric_detects_one_large_connected_core(self):
        shape = (80, 80)
        mask = _rect_mask(shape, 10, 10, 70, 70)
        frame = np.full((*shape, 3), (30, 60, 140), dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (49, 49), (255, 255, 255), -1)

        metrics = tip_runtime._specular_instance_metrics(frame, mask)

        self.assertGreater(metrics["specular_ratio"], 0.20)
        self.assertAlmostEqual(
            metrics["largest_specular_area"], metrics["specular_area"]
        )


if __name__ == "__main__":
    unittest.main()
