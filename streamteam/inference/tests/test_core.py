# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time as pytime
import copy

# Third-party
import emcee
import numpy as np
import pytest
import astropy.units as u
from astropy.io.misc import fnpickle
import matplotlib.pyplot as plt

from ..core import *
from ..parameter import *
from ..prior import *

plot_path = "plots/tests/inference"
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

class TestModelParameter(object):

    def test_string(self):
        p = ModelParameter("m", value=np.nan, truth=1.5,
                           prior=LogUniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm' truth=1.5>"
        assert str(p) == "m"

        p = ModelParameter("m", value=np.nan,
                           prior=LogUniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm'>"
        assert str(p) == "m"

class TestEmceeModel(object):

    def setup(self):
        np.random.seed(42)

        self.flat_model = EmceeModel()
        for name in "abcdefg":
            p = ModelParameter(name, value=np.nan, truth=np.random.random(),
                               prior=LogUniformPrior(0.,1.))
            self.flat_model.add_parameter(p)

        self.group_model = EmceeModel()
        for group in ["herp","derp"]:
            for name in "abcd":
                p = ModelParameter(name, value=np.nan, truth=np.random.random(),
                                   prior=LogUniformPrior(0.,1.))
                self.group_model.add_parameter(p, parameter_group=group)

        self.vec_model = EmceeModel()
        for name in "abcd":
            length = np.random.randint(2,10)
            troof = np.random.random(size=length)
            p = ModelParameter(name, value=[np.nan]*length, truth=troof,
                               prior=LogUniformPrior(0*troof,0*troof+1))
            self.vec_model.add_parameter(p)

        self.models = [self.group_model, self.flat_model, self.vec_model]

    def test_init(self):
        m = ModelParameter("m", value=np.nan, truth=1.5,
                           prior=LogUniformPrior(1.,2.))
        b = ModelParameter("b", value=np.nan, truth=6.7,
                           prior=LogUniformPrior(0.,10.))

        model = EmceeModel()
        model.add_parameter(m)
        model.add_parameter(b)
        model.parameters['main']['m']
        model.parameters['main']['b']

        assert np.all(model.truths == np.array([1.5,6.7]))

    def test_walk_parameters(self):
        for model in self.models:
            for group,name,p in model._walk(model.parameters):
                assert name == str(p)

    def test_decompose_compose(self):
        for model in self.models:
            vec = np.random.random(size=model.nparameters)
            decom = model.vector_to_parameters(vec)
            for group,name,p in model._walk(model.parameters):
                decom[group][name]

            com = model.parameters_to_vector(decom)
            assert np.all((vec-com) == 0.)

    def test_flatchain(self):
        nsteps = 1024
        for model in self.models:
            vec = np.random.random(size=(model.nparameters,nsteps))
            decom = model.vector_to_parameters(vec)
            for group,name,p in model._walk(model.parameters):
                print(decom[group][name].shape)

    def test_prior(self):
        for model in self.models:
            vec = np.random.random(size=model.nparameters)
            decom = model.vector_to_parameters(vec)
            print(model.ln_prior(decom))

    def test_likelihood(self):
        for model in self.models:
            vec = np.random.random(size=model.nparameters)
            decom = model.vector_to_parameters(vec)
            print(model.ln_likelihood(decom))

    def test_truths(self):
        for model in self.models:
            truths = np.array([])
            for group,name,p in model._walk(model.parameters):
                truths = np.append(truths,p.truth)

            assert np.all(truths == model.truths)

    def test_sample_priors(self):
        for model in self.models:
            pri = model.sample_priors(size=5)
            print(pri.shape)

    def test_mcmc_sample_priors(self):
        m = ModelParameter("m", value=np.nan, truth=1.,
                           prior=LogNormalPrior(0.,2.))
        b = ModelParameter("b", value=np.nan, truth=6.7,
                           prior=LogUniformPrior(0.,10.))

        model = EmceeModel()
        model.add_parameter(m)
        model.add_parameter(b)
        model.parameters['main']['m']
        model.parameters['main']['b']

        nwalkers = 16
        ndim = 2
        sampler = emcee.EnsembleSampler(nwalkers, ndim, model)

        p0 = [np.random.rand(ndim) for i in range(nwalkers)]
        sampler.run_mcmc(p0, 1000)

        fig,axes = plt.subplots(1,2)
        axes[0].hist(sampler.flatchain[:,0])
        axes[1].hist(sampler.flatchain[:,1])
        fig.savefig(os.path.join(plot_path,"priors.png"))

    def test_fit_line(self):
        m = ModelParameter("m", value=np.nan, truth=1.,
                           prior=LogUniformPrior(0.,2.))
        b = ModelParameter("b", value=np.nan, truth=6.7,
                           prior=LogUniformPrior(0.,10.))

        ndata = 15
        x = np.random.uniform(0.,10.,size=ndata)
        x.sort()
        y = m.truth*x + b.truth
        sigma_y = np.random.uniform(0.5,1.,size=ndata)
        y += np.random.normal(0., sigma_y)

        def ln_likelihood(parameters, x, y, sigma_y):
            model_val = parameters['line']['m']*x + parameters['line']['b']
            return -0.5*((y - model_val) / sigma_y)**2

        model = EmceeModel(ln_likelihood, (x,y,sigma_y))
        model.add_parameter(m, 'line')
        model.add_parameter(b, 'line')

        nwalkers = 16
        ndim = 2
        sampler = emcee.EnsembleSampler(nwalkers, ndim, model)

        p0 = model.sample_priors(size=nwalkers)
        sampler.run_mcmc(p0, 1000)

        fig,axes = plt.subplots(1,2)
        axes[0].hist(sampler.flatchain[500:,0])
        axes[1].hist(sampler.flatchain[500:,1])
        fig.savefig(os.path.join(plot_path,"fit_line.png"))
