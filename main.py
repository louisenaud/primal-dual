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


def nabla(I):
    """
    Function to compute the forward gradient of an image
    :param I: 
    :return: 
    """
    h, w = I.shape
    G = np.zeros((h, w, 2), I.dtype)
    G[:, :-1, 0] -= I[:, :-1]
    G[:, :-1, 0] += I[:, 1:]
    G[:-1, :, 1] -= I[:-1]
    G[:-1, :, 1] += I[1:]
    return G


def nablaT(G):
    """
    Function to compute the transpose of the forward gradient on an image
    :param G: 
    :return: 
    """
    h, w = G.shape[:2]
    I = np.zeros((h, w), G.dtype)
    # note that we just reversed left and right sides
    # of each line to obtain the transposed operator
    I[:, :-1] -= G[:, :-1, 0]
    I[:, 1: ] += G[:, :-1, 0]
    I[:-1]    -= G[:-1, :, 1]
    I[1: ]    += G[:-1, :, 1]
    return I

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
    
    :param I: numpy array [NxMx2], array with the same dimensions as the gradient of the image to denoise.
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
    div = d_h + d_v
    return div


def anorm(x):
    '''Calculate L2 norm over the last array dimention'''
    return np.sqrt((x * x).sum(-1))


def calc_energy_ROF(X, observation, clambda):
    Ereg = anorm(nabla(X)).sum()
    Edata = 0.5 * clambda * ((X - observation) ** 2).sum()
    return Ereg + Edata


def calc_energy_TVL1(X, observation, clambda):
    Ereg = anorm(nabla(X)).sum()
    Edata = clambda * np.abs(X - observation).sum()
    return Ereg + Edata


def project_nd(P, r):
    '''perform a pixel-wise projection onto R-radius balls'''
    nP = np.maximum(1.0, anorm(P) / r)
    return P / nP[..., np.newaxis]


def shrink_1d(X, F, step):
    '''pixel-wise scalar srinking'''
    return X + np.clip(F - X, -step, step)

def solve_ROF(img, clambda, iter_n=101):
    # setting step sizes and other params
    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2*tau)
    theta = 1.0

    X = img.copy()
    P = nabla(X)
    for i in xrange(iter_n):
        P = project_nd( P + sigma*nabla(X), 1.0 )
        lt = clambda * tau
        X1 = (X - tau * nablaT(P) + lt * img) / (1.0 + lt)
        X = X1 + theta * (X1 - X)
        if i % 10 == 0:
            print "%.2f" % calc_energy_ROF(X, img, clambda),
    print
    return X


if __name__ == '__main__':
    # Create image to noise and denoise
    img_ref = np.array(face(True))
    img_obs = skimage.util.random_noise(img_ref, mode='gaussian')

    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img_ref)
    ax1.set_title('Sharing Y axis')
    ax2.imshow(img_obs)
    plt.show()

    g1 = nabla(img_obs)
    g2 = forward_gradient(img_obs)
    i1 = nablaT(g1)
    i2 = backward_divergence(g2)

    # some reasonable lambdas
    lambda_ROF = 1.0
    lambda_TVL1 = 1.0

    print "ROF:",
    print calc_energy_ROF(img_obs, img_obs, lambda_ROF),
    print calc_energy_ROF(img_ref, img_obs, lambda_ROF)
    print "TV-L1:",
    print calc_energy_TVL1(img_obs, img_obs, lambda_TVL1),
    print calc_energy_TVL1(img_ref, img_obs, lambda_TVL1)

    img_denoised = solve_ROF(img_obs, 8.0)



    # row and column sharing
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    ax1.imshow(img_ref)
    ax1.set_title("Reference image")
    ax2.imshow(img_obs)
    ax2.set_title("Observed image")
    ax3.imshow(img_denoised)
    ax3.set_title("Denoised image")
    plt.show()
