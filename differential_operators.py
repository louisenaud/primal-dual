"""
Project:    
File:       differential_operators.py
Created by: louise
On:         8/22/17
At:         4:10 PM
"""

import numpy as np


def forward_gradient(im):
    """
    Function to compute the forward gradient of the image I.
    Definition from: http://www.ipol.im/pub/art/2014/103/, p208
    :param im: numpy array [MxN], input image
    :return: numpy array [MxNx2], gradient of the input image, the first channel is the horizontal gradient, the second 
    is the vertical gradient. 
    """
    h, w = im.shape
    gradient = np.zeros((h, w, 2), im.dtype)  # Allocate gradient array
    # Horizontal direction
    gradient[:, :-1, 0] = im[:, 1:] - im[:, :-1]
    # Vertical direction
    gradient[:-1, :, 1] = im[1:, :] - im[:-1, :]

    return gradient


def backward_divergence(grad):
    """
    Function to compute the backward divergence.
    Definition in : http://www.ipol.im/pub/art/2014/103/, p208

    :param grad: numpy array [NxMx2], array with the same dimensions as the gradient of the image to denoise.
    :return: numpy array [NxM], backward divergence 
    """

    h, w = grad.shape[:2]
    div = np.zeros((h, w), grad.dtype)  # Allocate divergence array
    # Horizontal direction
    d_h = np.zeros((h, w), grad.dtype)
    d_h[:, 0] = grad[:, 0, 0]
    d_h[:, 1:-1] = grad[:, 1:-1, 0] - grad[:, :-2, 0]
    d_h[:, -1] = -grad[:, -2:-1, 0].flatten()

    # Vertical direction
    d_v = np.zeros((h, w), grad.dtype)
    d_v[0, :] = grad[0, :, 1]
    d_v[1:-1, :] = grad[1:-1, :, 1] - grad[:-2, :, 1]
    d_v[-1, :] = -grad[-2:-1, :, 1].flatten()

    # Divergence
    div = d_h + d_v
    return div
