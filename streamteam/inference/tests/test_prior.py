# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime
import copy

# Third-party
import numpy as np
import pytest
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.stats import norm

from ..prior import *

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

def test_uniform():

    prior = LogUniformPrior(0.,1.)
    assert prior(0.5) == 0.
    assert prior(1.5) == -np.inf
    assert prior.sample(n=10).shape == (10,)

    prior = LogUniformPrior([0.,1],[1.,2])
    assert np.all(prior([0.5,1.5]) == np.array([0., 0.]))
    assert np.all(prior([1.5,0.5]) == np.array([-np.inf, -np.inf]))
    assert np.all(prior([1.5,1.5]) == np.array([-np.inf, 0.]))
    assert prior.sample(n=10).shape == (10,2)

def test_normal1d():
    prior = LogNormal1DPrior(mean=0., stddev=0.5)
    assert prior.sample(11).shape == (11,)
    assert np.allclose(prior(0.2), np.log(norm.pdf(0.2, loc=0., scale=0.5)))

    plt.clf()
    plt.hist(np.squeeze(prior.sample(1024)))
    plt.savefig(os.path.join(plot_path,"norm_hist1.png"))

    # 2 1D Gaussians
    prior = LogNormal1DPrior(mean=[0.,1.214],
                             stddev=[0.5,0.6])
    one = np.log(norm.pdf(0.2, loc=0., scale=0.5))
    two = np.log(norm.pdf(1.74, loc=1.214, scale=0.6))
    assert np.allclose(prior([0.2, 1.74]), [one,two])
    assert prior.sample(11).shape == (11,2)

    plt.clf()
    s = prior.sample(1024)
    plt.hist(s[:,0])
    plt.hist(s[:,1])
    plt.savefig(os.path.join(plot_path,"norm_hist2.png"))

# def test_normalnd():
#     return
#     # 1 2D Gaussian
#     prior = LogNormalPrior(mean=np.array([[0.,1.]]),
#                            stddev=np.array([[0.5,0.5]]))
#     print(prior([[0.2, 1.2]]))
#     assert prior.shape == (1,2)
#     assert prior.cov.shape == (1,2,2)
#     assert prior._norm.shape == (1,)
#     assert prior.sample(11).shape == (11,1,2)
#     plt.clf()
#     s = prior.sample(1024)
#     plt.plot(s[:,0,0], s[:,0,1], linestyle='none')
#     plt.xlim(-2.,3.); plt.ylim(-2.,3.)
#     plt.savefig(os.path.join(plot_path,"norm_hist3.png"))

#     # 1 2D Gaussian
#     prior = LogNormalPrior(mean=np.array([[0.,1.]]),
#                            cov=np.array([[0.25,0.],[0.,0.25]]))
#     print(prior([[0.2, 1.2]]))
#     assert prior.shape == (1,2)
#     assert prior.cov.shape == (1,2,2)
#     assert prior.stddev is None
#     assert prior._norm.shape == (1,)
#     assert prior.sample(11).shape == (11,1,2)
#     plt.clf()
#     s = prior.sample(1024)
#     plt.plot(s[:,0,0], s[:,0,1], linestyle='none')
#     plt.xlim(-2.,3.); plt.ylim(-2.,3.)
#     plt.savefig(os.path.join(plot_path,"norm_hist4.png"))

#     # 1 rotated 2D Gaussian
#     prior = LogNormalPrior(mean=np.array([[0.,1.]]),
#                            cov=np.array([[0.25,-0.17],[-0.17,0.25]]))
#     print(prior([[0.2, 1.2]]))
#     assert prior.shape == (1,2)
#     assert prior.cov.shape == (1,2,2)
#     assert prior.stddev is None
#     assert prior._norm.shape == (1,)
#     assert prior.sample(11).shape == (11,1,2)
#     plt.clf()
#     s = prior.sample(1024)
#     plt.plot(s[:,0,0], s[:,0,1], linestyle='none')
#     plt.xlim(-2.,3.); plt.ylim(-2.,3.)
#     plt.savefig(os.path.join(plot_path,"norm_hist5.png"))