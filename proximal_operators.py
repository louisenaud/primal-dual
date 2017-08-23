"""
Project:    
File:       proximal_operators.py
Created by: louise
On:         8/22/17
At:         4:39 PM
"""

import numpy as np
from norm import norm1, norm2


def proximal_linf_ball(p, r=1.0):
    """
    Proximal operator for sum(gradient(x)).
    :param p: numpy array [MxNx2], 
    :param r: float, radius of infinity norm ball.
    :return: numpy array, same dimensions as p
    """
    n_p = np.maximum(1.0, norm2(p) / r)
    return p / n_p[..., np.newaxis]


def proximal_l1(x, f, clambda):
    """

    :param x: numpy array, [MxN], primal variable,
    :param f: numpy array, [MxN], observed image,
    :param clambda: float, parameter for data term.
    :return: numpy array, [MxN]
    """
    return x + np.clip(f - x, -clambda, clambda)
