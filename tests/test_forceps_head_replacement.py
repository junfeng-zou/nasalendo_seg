"""Ensure missed detections cannot inflate reported classification performance."""
import unittest
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location("head_replacement", Path(__file__).resolve().parents[1] / "scripts/evaluate_forceps_head_replacement.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
metrics, iou = module.metrics, module.iou

class ReplacementMetricsTests(unittest.TestCase):
    def test_miss_is_false_negative_not_removed(self):
        rows=[dict(label='TooFar',auto_prediction='TooFar'),dict(label='Good',auto_prediction='Invalid'),dict(label='TooClose',auto_prediction='TooClose')]
        m=metrics(rows,'auto_prediction')
        self.assertEqual(m['n'],3)
        self.assertAlmostEqual(m['accuracy'],2/3)
        self.assertAlmostEqual(m['macro_f1'],2/3)
        self.assertEqual(m['confusion'][1],[0,0,0,1])
    def test_wrong_class_counts_as_false_positive_and_negative(self):
        rows=[dict(label='TooFar',auto_prediction='Good'),dict(label='Good',auto_prediction='Good'),dict(label='TooClose',auto_prediction='TooClose')]
        m=metrics(rows,'auto_prediction')
        self.assertAlmostEqual(m['macro_f1'],(0+2/3+1)/3)
    def test_no_detection_is_not_overlap(self):
        self.assertEqual(iou([0,0,10,10],None),0)
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(iou([0,0,10,10],[10,10,20,20]),0)
if __name__=='__main__':unittest.main()
