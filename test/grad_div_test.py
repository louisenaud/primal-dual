"""
Project:    
File:       grad_div_test.py
Created by: louise
On:         8/22/17
At:         2:37 PM
"""
import unittest
import numpy as np
from scipy.misc import face
from main import forward_gradient, backward_divergence


class TestAdjoint(unittest.TestCase):

    def test_adjoint_operator(self):
        Y = 200
        X = 100
        x = 1 + np.random.randn(Y, X)
        y_l = np.zeros((Y + 1, X + 1, 2))

        y_l[1:, 1:-1, 0] = 1 + np.random.randn(Y, X - 1)
        y_l[1:-1, 1:, 1] = 1 + np.random.randn(Y - 1, X)
        y = y_l[1:, 1:, :]
        print y.shape
        # Compute gradient and divergence
        gx = forward_gradient(x)
        dy = backward_divergence(y)

        check = abs((y[:] * gx[:]).sum() + (dy[:]*x[:]).sum())

        self.assertAlmostEquals(0, check)
