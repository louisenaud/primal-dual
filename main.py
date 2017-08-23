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

from norm import norm1, norm2
from differential_operators import backward_divergence, forward_gradient
from proximal_operators import proximal_linf_ball


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
    Compute the dual energy of ROF problem.
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
    energy_reg = norm1(forward_gradient(x)).sum()
    energy_data_term = 0.5*clambda * norm2(x - img_obs).sum()
    return energy_reg + energy_data_term


def primal_energy_tvl1(x, observation, clambda):
    """

    :param X: 
    :param observation: 
    :param clambda: 
    :return: 
    """
    energy_reg = norm1(forward_gradient(x)).sum()
    energy_data_term = clambda * np.abs(x - observation).sum()
    return energy_reg + energy_data_term




if __name__ == '__main__':
    # Create image to noise and denoise
    img_ref = np.array(face(True))
    img_obs = skimage.util.random_noise(img_ref, mode='gaussian')
    g2 = forward_gradient(img_obs)
    i2 = backward_divergence(g2)

    norm_l = 7.0
    max_it = 3000
    theta = 1.0
    alpha = 0.1
    w_sig = 10.0
    L2 = 8.0
    tau = 0.01
    sigma = 1.0 / (norm_l * tau)
    lambda_TVL1 = 1.0
    lambda_rof = 7.0

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
        y = proximal_linf_ball(y, 1.0)
        # Primal update
        x_old = x
        x = (x + tau * backward_divergence(y) + lambda_rof * tau * img_obs) / (1.0 + lambda_rof * tau)
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

