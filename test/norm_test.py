"""
Project:    
File:       norm_test.py
Created by: louise
On:         8/22/17
At:         3:03 PM
"""

import unittest
import numpy as np
from norm import norm1, norm2


class TestNorm(unittest.TestCase):

    def test_norm(self):
        self.assertEqual(0, norm1(np.array([0, 0, 0, 0])))
        self.assertEqual(0, norm2(np.array([0, 0, 0, 0])))
        self.assertEqual(1, norm1(np.array([0, 1, 0, 0])))
        self.assertEqual(1, norm2(np.array([0, 0, 1, 0])))
