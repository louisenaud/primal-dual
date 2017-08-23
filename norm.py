"""
Project:    
File:       norm.py
Created by: louise
On:         8/22/17
At:         2:56 PM
"""

import numpy as np


def norm2(x):
    """
    Computes the L2 norm of a vector x.
    :param x: numpy array
    :return: float
    """
    return np.sqrt((x ** 2).sum(-1))


def norm1(x):
    """
    Computes the L1 norm of a vector x.
    :param x: numpy array
    :return: float
    """
    return np.abs(x).sum(-1)