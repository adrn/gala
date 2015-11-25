# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from ..prior import *

def test_uniform():

    prior = UniformPrior(0.,1.)
    prior.logpdf(0.5)
    assert prior.logpdf(1.5) == -np.inf
    assert prior.sample(n=10).shape == (10,)

    prior = UniformPrior([0.,1],[1.,2])
    np.all(prior.logpdf([0.5,1.5]))
    assert np.all(prior.logpdf([1.5,0.5]) == np.array([-np.inf, -np.inf]))
    assert np.all(prior.logpdf([1.5,1.5]) == np.array([-np.inf, 0.]))
    assert prior.sample(n=10).shape == (10,2)

def test_logarithmic():

    prior = LogarithmicPrior(1.,2.)
    assert prior.logpdf(1.5) == np.log(1/np.log(2/1.))
    assert prior.logpdf(0.5) == -np.inf
    assert prior.sample(n=10).shape == (10,)

    prior = LogarithmicPrior([0.,1],[1.,2])
    np.all(prior.logpdf([0.5,1.5]))
    assert np.all(prior.logpdf([1.5,0.5]) == np.array([-np.inf, -np.inf]))
    assert np.all(prior.logpdf([1.5,1.5]) == np.array([-np.inf, np.log(1./np.log(2/1.))]))
    assert prior.sample(n=10).shape == (10,2)

def test_normal(tmpdir):
    prior = NormalPrior(mean=0., stddev=0.5)
    assert prior.sample(11).shape == (11,)
    assert np.allclose(prior.logpdf(0.2), norm.logpdf(0.2, loc=0., scale=0.5))

    plt.clf()
    plt.hist(np.squeeze(prior.sample(1024)))
    plt.savefig(os.path.join(str(tmpdir),"norm_hist1.png"))

    # 2 independent Gaussians
    prior = NormalPrior(mean=[0.,1.214],
                        stddev=[0.5,0.6])
    one = norm.logpdf(0.2, loc=0., scale=0.5)
    two = norm.logpdf(1.74, loc=1.214, scale=0.6)
    assert np.allclose(prior.logpdf([0.2, 1.74]), [one,two])
    assert prior.sample(11).shape == (11,2)

    plt.clf()
    s = prior.sample(1024)
    plt.hist(s[:,0])
    plt.hist(s[:,1])
    plt.savefig(os.path.join(str(tmpdir),"norm_hist2.png"))
