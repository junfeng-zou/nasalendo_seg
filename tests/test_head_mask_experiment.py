"""Verify that mask guidance retains background and never moves image pixels."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_head_mask_experiment import soft_mask

class SoftMaskTests(unittest.TestCase):
    def test_foreground_is_preserved_exactly(self):
        rng=np.random.default_rng(42)
        image=rng.integers(0,256,(13,17,3),dtype=np.uint8)
        np.testing.assert_array_equal(soft_mask(image,np.full((13,17),255,np.uint8)),image)
    def test_missing_mask_retains_dimmed_rgb(self):
        image=np.array([[[101,200,255],[40,80,120]]],dtype=np.uint8)
        expected=np.rint(image.astype(np.float32)*.5).astype(np.uint8)
        np.testing.assert_array_equal(soft_mask(image,np.zeros((1,2),np.uint8)),expected)
    def test_mask_boundary_keeps_coordinates_and_channels(self):
        image=np.array([[[20,40,60],[80,100,120]]],dtype=np.uint8)
        np.testing.assert_array_equal(soft_mask(image,np.array([[255,0]],np.uint8)),np.array([[[20,40,60],[40,50,60]]],np.uint8))
if __name__=='__main__':unittest.main()
