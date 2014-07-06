# Emcee Model/Parameter API sketch

# Over-arching question about Parameters:
#   - Do we include a value in the Parameter object, or should that be kept separate?
#   - One option is yes, and ln_posterior has to update the value's of all of the
#       parameter objects for each step. Parameter's must be picklable and thread-safe
#       for this to work with multiprocessing
#   - Other option is no, and what is passed around the model is a dictionary with
#       key = parameter object, value = numerical value.
#   - For now, I'm going to go with option 2 but we can always modify this

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
from astropy import log as logger
import astropy.units as u
import emcee
import matplotlib.pyplot as plt
import numpy as np

# Project
from ..core import EmceeModel
from ..parameter import ModelParameter
from ..prior import LogUniformPrior, LogNormalPrior

# Example:
def ln_prior(parameters, value_dict, *args):
    # parameters can be a hierarchical collection
    return 0.

def ln_likelihood(parameters, value_dict, *args):
    x,y,sigma_y = args
    m = value_dict["m"]
    b = value_dict["b"]
    return -0.5 * (y - m*x - b)**2 / sigma_y**2

def test():
    m = ModelParameter("m", truth=2.1,
                       prior=LogUniformPrior(a=0., b=5.))
    b = ModelParameter("b", truth=1.,
                       prior=LogUniformPrior(a=0., b=5.))

    # generate data
    x = np.random.uniform(0., 15., size=5)
    sigma_y = np.random.uniform(0.1, 0.3, size=x.size)
    y = m.truth*x + b.truth

    model = EmceeModel(ln_likelihood, ln_prior=None,
                       args=(x,y,sigma_y))
    model.add_parameter(m)
    model.add_parameter(b)

    assert model.parameters.has_key("m")
    assert model.parameters.has_key("b")

    left = model.ln_posterior(dict(m=2.1, b=0.5))
    true = model.ln_posterior(dict(m=2.1, b=1.))
    right = model.ln_posterior(dict(m=2.1, b=1.5))
    assert left < true and right < true

    print(model.truth_vector)

    # sample from priors
    nwalkers = 16
    p0 = model.sample_priors(size=nwalkers)

    # define sampler object
    sampler = emcee.EnsembleSampler(nwalkers, model.nparameters, model)
    pos,prob,state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    pos,prob,state = sampler.run_mcmc(pos, 200)

    plt.subplot(121)
    plt.hist(sampler.flatchain[:,0])
    plt.axvline(m.truth.value, color='b')
    plt.xlim(m.prior.a, m.prior.b)

    plt.subplot(122)
    plt.hist(sampler.flatchain[:,1])
    plt.axvline(b.truth.value, color='b')
    plt.xlim(b.prior.a, b.prior.b)

    plt.show()

def test_wrong_prior():
    m = ModelParameter("m", truth=2.1,
                       prior=LogNormalPrior(mean=1.1, stddev=0.2))
    b = ModelParameter("b", truth=1.,
                       prior=LogUniformPrior(a=0., b=5.))

    # generate data
    x = np.random.uniform(0., 15., size=10) # change size
    sigma_y = np.random.uniform(0.5, 1., size=x.size)
    y = m.truth*x + b.truth

    model = EmceeModel(ln_likelihood, ln_prior=None,
                       args=(x,y,sigma_y))
    model.add_parameter(m)
    model.add_parameter(b)

    assert model.parameters.has_key("m")
    assert model.parameters.has_key("b")

    left = model.ln_posterior(dict(m=2.1, b=0.5))
    true = model.ln_posterior(dict(m=2.1, b=1.))
    right = model.ln_posterior(dict(m=2.1, b=1.5))
    assert left < true and right < true

    print(model.truth_vector)

    # sample from priors
    nwalkers = 16
    p0 = model.sample_priors(size=nwalkers)

    # define sampler object
    sampler = emcee.EnsembleSampler(nwalkers, model.nparameters, model)
    pos,prob,state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    pos,prob,state = sampler.run_mcmc(pos, 200)

    plt.subplot(121)
    plt.hist(sampler.flatchain[:,0], normed=True)
    plt.hist(m.prior.sample(size=1000), alpha=0.5, normed=True)
    plt.axvline(m.truth.value, color='b')
    #plt.xlim(m.prior.a, m.prior.b)

    plt.subplot(122)
    plt.hist(sampler.flatchain[:,1])
    plt.axvline(b.truth.value, color='b')
    plt.xlim(b.prior.a, b.prior.b)

    plt.show()