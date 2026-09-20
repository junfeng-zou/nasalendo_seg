import unittest
import numpy as np
import torch
from distance_state_classifier.src.manual_head_abc import (
    parse_head_box, prepare_inputs, letterbox, fit_scale_normalizer, FusionHead)


class ManualHeadTests(unittest.TestCase):
    def test_two_and_four_point_annotations_agree(self):
        def annotation(points):
            return {'shapes': [{'label': 'forceps_head', 'shape_type': 'rectangle', 'points': points}]}
        a = parse_head_box(annotation([[10, 20], [30, 60]]), 100, 100)
        b = parse_head_box(annotation([[10, 20], [30, 20], [30, 60], [10, 60]]), 100, 100)
        self.assertEqual(a, b)
        self.assertIsNone(parse_head_box({'shapes': []}, 100, 100))
        with self.assertRaises(ValueError):
            parse_head_box(annotation([[-1, 0], [20, 20]]), 100, 100)

    def test_letterbox_preserves_aspect(self):
        result = letterbox(np.full((20, 40, 3), 255, np.uint8), 100)
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertTrue((result[25:75] == 255).all())
        self.assertTrue((result[:25] == 0).all())

    def test_original_scale_is_independent_of_crop_and_output_size(self):
        image = np.full((60, 80, 3), 100, np.uint8)
        _, local, scales = prepare_inputs(image, [0, 0, 20, 40], 100, 64, 1.2)
        _, _, other = prepare_inputs(image, [0, 0, 20, 40], 100, 128, 1.5)
        np.testing.assert_allclose(scales, [.2, .4])
        np.testing.assert_array_equal(scales, other)
        self.assertEqual(local.shape, (64, 64, 3))
        self.assertTrue((local == 0).any())

    def test_validation_values_do_not_affect_normalizer(self):
        a = np.array([[1., 2.], [3., 4.], [100., 200.]])
        b = a.copy(); b[-1] = [-1000., 3000.]
        for first, second in zip(fit_scale_normalizer(a, [1, 1, 0]), fit_scale_normalizer(b, [1, 1, 0])):
            np.testing.assert_array_equal(first, second)

    def test_equal_head_initialization_and_scale_gradient(self):
        torch.manual_seed(42); first = FusionHead()
        torch.manual_seed(42); second = FusionHead()
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))
        scales = torch.tensor([[.2, -.5], [.8, .4]], requires_grad=True)
        loss = first(torch.randn(2, 768), scales).square().mean()
        loss.backward()
        self.assertGreater(float(scales.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
