"""
Project:    primal-dual
File:       main.py
Created by: louise
On:         7/6/17
At:         2:11 PM
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy.misc import face
import multiprocessing


def forward_gradient(im):
    """
    Function to compute the forward gradient of the image I.
    Definition from: http://www.ipol.im/pub/art/2014/103/, p208
    :param I: 
    :return: 
    """
    h, w = im.shape
    # Gradient
    gradient = np.zeros((h, w, 2), im.dtype)
    gradient[:, :-1, 0] = im[:, 1:] - im[:, :-1]
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
    # Horizontal direction
    d_h = np.zeros((h, w), grad.dtype)
    d_h[:, 0] = grad[:, 0, 0]
    d_h[:, 1:-1] = grad[:, 1:-1, 0] - grad[:, :-2, 0]
    d_h[:, -1:] = -grad[:, -2:-1, 0]

    # Vertical direction
    d_v = np.zeros((h, w), grad.dtype)
    d_v[0, :] = grad[0, :, 1]
    d_v[1:-1, :] = grad[1:-1, :, 1] - grad[:-2, :, 1]
    d_v[-1:, :] = -grad[-2:-1, :, 1]

    # Divergence
    div = np.zeros((h, w), grad.dtype)
    div = d_h + d_v
    return div


def norm2(x):
    """
    L2 norm.
    :param x: 
    :return: 
    """
    return np.sqrt((x**2).sum(-1))


def norm1(x):
    """
    L1 norm.
    :param x: 
    :return: 
    """
    return np.abs(x).sum(-1)


def dual_energy_tvl1(y, im_obs):
    """
    Compute the dual energy of TV-L1 problem.
    :param y: 
    :param im_obs: numpy array, observed image
    :return: float, dual energy
    """
    nrg = -0.5 * (im_obs - backward_divergence(y))**2
    nrg = np.sum(nrg)
    return nrg


def dual_energy_rof(y, im_obs):
    """
    Compute the dual energy of TV-L1 problem.
    :param y: 
    :param im_obs: numpy array, observed image
    :return: float, dual energy
    """
    nrg = -0.5 * (im_obs - backward_divergence(y))**2
    nrg = nrg.sum()
    return nrg


def primal_energy_rof(x, img_obs, clambda):
    """

    :param x: 
    :param img_obs: 
    :param clambda: 
    :return: 
    """
    g = forward_gradient(x)
    nrg = wp * np.maximum(g, 0) + wn * np.maximum(-g, 0)
    a = np.matmul(np.transpose(x), x)
    b = np.matmul(np.transpose(img_obs), x)
    nrg = 0.5* a.sum() + b.sum() + nrg.sum()
    #energy_reg = norm1(forward_gradient(x)).sum()
    #energy_data_term = 0.5 * clambda * ((x - img_obs) ** 2).sum()
    return nrg


def primal_energy_tvl1(X, observation, clambda):
    """

    :param X: 
    :param observation: 
    :param clambda: 
    :return: 
    """
    energy_reg = norm2(forward_gradient(X)).sum()
    energy_data_term = clambda * np.abs(X - observation).sum()
    return energy_reg + energy_data_term


def proximal_linf_ball(p, r):
    """
    
    :param p: 
    :param r: 
    :return: 
    """
    n_p = np.maximum(1.0, norm2(p) / r)
    return p / n_p[..., np.newaxis]


def proximal_l1(x, f, clambda):
    """
    
    :param x: 
    :param f: 
    :param clambda: 
    :return: 
    """
    return x + np.clip(f - x, -clambda, clambda)


if __name__ == '__main__':
    # Create image to noise and denoise
    img_ref = np.array(face(True))
    img_obs = skimage.util.random_noise(img_ref, mode='gaussian')

    g2 = forward_gradient(img_obs)
    i2 = backward_divergence(g2)

    norm_l = 2.5
    max_it = 101
    theta = 1.0
    alpha = 0.1
    w_sig = 10.0
    L2 = 8.0
    tau = 0.01
    sigma = 1.0 / (norm_l * tau)
    lambda_TVL1 = 1.0
    lambda_rof = 1.5

    x = img_obs
    x_tilde = x
    h, w = img_ref.shape
    y = np.zeros((h, w, 2))
    wn = w_sig * np.random.random((h, w, 2))
    wp = wn

    p_nrg = primal_energy_rof(x, img_obs, lambda_TVL1)
    print "Primal Energy = ", p_nrg
    d_nrg = dual_energy_rof(y, img_obs)
    print "Dual Energy = ", d_nrg

    # Solve ROF
    primal = np.zeros((max_it,))
    dual = np.zeros((max_it,))
    gap = np.zeros((max_it,))
    primal[0] = p_nrg
    dual[0] = d_nrg
    y = forward_gradient(x)
    for it in range(max_it):
        # Dual update
        y = y + sigma * forward_gradient(x_tilde)
        #y = np.maximum(-wn, np.minimum(y, wp))
        y = proximal_linf_ball(y, 1.0)
        # Primal update
        x_old = x
        x = (x - tau * (img_obs - backward_divergence(y))) / (1.0 + tau)
        #lt = lambda_rof * tau
        #x = (x - tau * backward_divergence(y) + lt * img_obs) / (1.0 + lt)
        # Smoothing
        x_tilde = x + theta * (x - x_old)

        # Compute energies
        primal[it] = primal_energy_rof(x_tilde, img_obs, sigma)
        dual[it] = dual_energy_rof(y, img_obs)
        gap[it] = primal[it] - dual[it]

    plt.figure()
    plt.plot(np.asarray(range(max_it)), primal, label="Primal Energy")
    plt.legend()

    plt.figure()
    plt.plot(np.asarray(range(max_it)), dual, label="Dual Energy")
    plt.legend()

    plt.figure()
    plt.plot(np.asarray(range(max_it)), gap, label="Gap")
    plt.legend()

    # row and column sharing
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.imshow(img_ref)
    ax1.set_title("Reference image")
    ax2.imshow(img_obs)
    ax2.set_title("Observed image")
    ax3.imshow(x)
    ax3.set_title("Denoised image")
    plt.show()

