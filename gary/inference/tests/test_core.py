# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import emcee
import numpy as np
import matplotlib.pyplot as plt

from ..model import *
from ..parameter import *
from ..prior import *

def dummy_likelihood(parameters, value_dict, *args):
    return 0.

class TestEmceeModel(object):

    def test_simple(self):
        m = ModelParameter("m", truth=1.5, prior=UniformPrior(1.,2.))
        b = ModelParameter("b", truth=6.7, prior=UniformPrior(0.,10.))

        model = EmceeModel(dummy_likelihood)
        model.add_parameter(m)
        model.add_parameter(b)
        model.parameters['m']
        model.parameters['b']

        assert np.all(model.truth_vector == np.array([1.5,6.7]))

    def setup(self):
        np.random.seed(42)

        self.flat_model = EmceeModel(dummy_likelihood)
        for name in "abcdefg":
            p = ModelParameter(name, truth=np.random.random(),
                               prior=UniformPrior(0.,1.))
            self.flat_model.add_parameter(p)
            assert self.flat_model.parameters.has_key(name)

        # ---------------------------------------------------------------
        self.group_model = EmceeModel(dummy_likelihood)
        for group in ["herp","derp"]:
            for name in "abcd":
                p = ModelParameter(name, truth=np.random.random(),
                                   prior=UniformPrior(0.,1.))
                self.group_model.add_parameter(p, group)
        assert self.group_model.parameters.has_key("herp")
        assert self.group_model.parameters.has_key("derp")

        # ---------------------------------------------------------------
        self.vec_model = EmceeModel(dummy_likelihood)
        for name in "abcd":
            troof = np.random.random(size=3)
            p = ModelParameter(name, truth=troof,
                               prior=UniformPrior(0*troof,0*troof+1))
            self.vec_model.add_parameter(p)
            assert self.vec_model.parameters.has_key(name)

        # ---------------------------------------------------------------
        self.frozen_model = EmceeModel(dummy_likelihood)
        for name in "abc":
            p = ModelParameter(name, truth=np.random.random(),
                               prior=UniformPrior(0.,1.))
            self.frozen_model.add_parameter(p)

        p = ModelParameter('mrfreeze', truth=0.5, prior=UniformPrior(0,1))
        p.frozen = 0.4712
        self.frozen_model.add_parameter(p)
        assert self.frozen_model.parameters['mrfreeze'].frozen is not False

        p = ModelParameter('chillout', truth=0.5, prior=UniformPrior(0,1))
        p.frozen = 0.4712
        self.frozen_model.add_parameter(p)
        assert self.frozen_model.parameters['chillout'].frozen is not False

        # ---------------------------------------------------------------

        self.models = [self.flat_model, self.group_model, self.vec_model, self.frozen_model]

    def test_walk_parameters(self):
        model_names = list("abcdefg")
        for group,name,p in self.flat_model._walk():
            assert name == str(p)
            model_names.pop(model_names.index(name))
        assert len(model_names) == 0

        model_names = list("abcdabcd")
        for group,name,p in self.group_model._walk():
            assert name == str(p)
            model_names.pop(model_names.index(name))
        assert len(model_names) == 0

        model_names = list("abcd")
        for group,name,p in self.vec_model._walk():
            assert name == str(p)
            model_names.pop(model_names.index(name))
        assert len(model_names) == 0

        model_names = list("abc")
        for group,name,p in self.frozen_model._walk():
            assert name == str(p)
            model_names.pop(model_names.index(name))
        assert len(model_names) == 0

    def test_nparameters(self):
        assert self.flat_model.nparameters == 7
        assert self.group_model.nparameters == 8
        assert self.vec_model.nparameters == 3*4
        assert self.frozen_model.nparameters == 3

    def test_devectorize_vectorize(self):

        for model in self.models:
            vec = np.random.random(size=model.nparameters)

            decom = model.devectorize(vec)
            for group,name,p in model._walk():
                if group is not None:
                    decom[group][name]
                else:
                    decom[name]

            com = model.vectorize(decom)
            assert np.all((vec-com) == 0.)

    def test_flatchain(self):
        nsteps = 1024
        for model in self.models:
            vec = np.random.random(size=(model.nparameters,nsteps))
            decom = model.devectorize(vec)
            for group,name,p in model._walk():
                if group is None:
                    print(decom[name].shape)
                else:
                    print(decom[group][name].shape)

    def test_prior(self):
        for model in self.models:
            vec = np.random.random(size=model.nparameters)
            decom = model.devectorize(vec)
            print(model.ln_prior(model.parameters, decom))

    def test_likelihood(self):
        for model in self.models:
            vec = np.random.random(size=model.nparameters)
            decom = model.devectorize(vec)
            print(model.ln_likelihood(model.parameters, decom))

    def test_sample_priors(self):
        for model in self.models:
            pri = model.sample_priors(n=5)

    def test_mcmc_sample_priors(self):
        m = ModelParameter("m", truth=1., prior=LogarithmicPrior(0.,2.))
        b = ModelParameter("b", truth=6.7, prior=UniformPrior(0.,10.))

        model = EmceeModel(dummy_likelihood)
        model.add_parameter(m)
        model.add_parameter(b)

        nwalkers = 16
        p0 = model.sample_priors(n=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, model.nparameters, model)
        sampler.run_mcmc(p0, 1000)


def line_likelihood(parameters, value_dict, x, y, sigma_y):
        try:
            m = value_dict['m']
        except KeyError:
            m = parameters['m'].frozen

        try:
            b = value_dict['b']
        except KeyError:
            b = parameters['b'].frozen

        model_val = m*x + b
        return -0.5*((y - model_val) / sigma_y)**2

class TestFitLine(object):

    def setup(self):
        m = ModelParameter("m", truth=1., prior=UniformPrior(0.,2.))
        b = ModelParameter("b", truth=6.7, prior=UniformPrior(0.,10.))

        ndata = 15
        x = np.random.uniform(0.,10.,size=ndata)
        x.sort()
        y = m.truth*x + b.truth
        sigma_y = np.random.uniform(0.5,1.,size=ndata)
        y += np.random.normal(0., sigma_y)

        self.model = EmceeModel(line_likelihood, args=(x,y,sigma_y))
        self.model.add_parameter(m)
        self.model.add_parameter(b)

    def test_vary_both(self):
        nwalkers = 16
        p0 = self.model.sample_priors(n=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,
                                        self.model.nparameters,
                                        self.model)
        pos,prob,state = sampler.run_mcmc(p0, 100)
        sampler.reset()
        pos,prob,state = sampler.run_mcmc(pos, 1000)

        fig,axes = plt.subplots(1,2)
        axes[0].hist(sampler.flatchain[:,0], normed=True)
        axes[0].hist(self.model.parameters['m'].prior.sample(n=1000),
                     alpha=0.5, zorder=-1, normed=True)
        axes[0].axvline(self.model.parameters['m'].truth.value, color='g')

        axes[1].hist(sampler.flatchain[:,1], normed=True)
        axes[1].hist(self.model.parameters['b'].prior.sample(n=1000),
                     alpha=0.5, zorder=-1, normed=True)
        axes[1].axvline(self.model.parameters['b'].truth.value, color='g')
        # fig.savefig(os.path.join(plot_path,"fit_line_vary_m_b.png"))

    def test_m_frozen(self):
        self.model.parameters['m'].freeze(self.model.parameters['m'].truth.value)

        nwalkers = 16
        p0 = self.model.sample_priors(n=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,
                                        self.model.nparameters,
                                        self.model)
        pos,prob,state = sampler.run_mcmc(p0, 100)
        sampler.reset()
        pos,prob,state = sampler.run_mcmc(pos, 1000)

        fig,ax = plt.subplots(1,1)
        ax.hist(sampler.flatchain[:,0], normed=True)
        ax.hist(self.model.parameters['b'].prior.sample(n=1000),
                alpha=0.5, zorder=-1, normed=True)
        ax.axvline(self.model.parameters['b'].truth.value, color='g')

        # fig.savefig(os.path.join(plot_path,"fit_line_vary_b.png"))

    def test_b_frozen(self):
        self.model.parameters['b'].freeze(self.model.parameters['b'].truth.value)

        nwalkers = 16
        p0 = self.model.sample_priors(n=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers,
                                        self.model.nparameters,
                                        self.model)
        pos,prob,state = sampler.run_mcmc(p0, 100)
        sampler.reset()
        pos,prob,state = sampler.run_mcmc(pos, 1000)

        fig,ax = plt.subplots(1,1)
        ax.hist(sampler.flatchain[:,0], normed=True)
        ax.hist(self.model.parameters['m'].prior.sample(n=1000),
                alpha=0.5, zorder=-1, normed=True)
        ax.axvline(self.model.parameters['m'].truth.value, color='g')

        # fig.savefig(os.path.join(plot_path,"fit_line_vary_m.png"))

def jvp_line_likelihood(parameters, value_dict, x, y):
    alpha = value_dict['alpha']
    beta = value_dict['beta']
    sigma = value_dict['sigma']

    # model_val = alpha + beta*x
    # return -0.5*(np.sum(np.log(2*np.pi*sigma**2) + ((y - model_val) / sigma)**2))

    y_model = alpha + beta * x
    return -0.5 * np.sum(np.log(2 *np.pi*sigma**2) + (y - y_model) ** 2 / sigma ** 2)

def ln_p_beta_sigma(parameters, value_dict, *args):
    beta = value_dict['beta']
    sigma = value_dict['sigma']

    if sigma < 0:
        return -np.inf
    else:
        return -1.5 * np.log(1 + beta**2) - np.log(sigma)

class TestJakeVDPExample(object):

    def setup(self):
        alpha = ModelParameter("alpha", truth=11., prior=UniformPrior(0.,20))
        beta = ModelParameter("beta", truth=2.7)
        sigma = ModelParameter("sigma", truth=0.25)

        ndata = 50
        x = np.random.uniform(0.,10.,size=ndata)
        x.sort()
        y = alpha.truth + beta.truth*x
        y += np.random.normal(0., sigma.truth, size=ndata)

        self.model = EmceeModel(jvp_line_likelihood, args=(x,y))
        self.model.add_parameter(alpha)
        self.model.add_parameter(beta)
        self.model.add_parameter(sigma)
        self.model.joint_priors.append(ln_p_beta_sigma)

    def test_sample(self):
        nwalkers = 50
        # p0 = emcee.utils.sample_ball([10., 5, 0.1], [0.1, 0.05, 0.01], size=nwalkers)
        p0 = np.random.uniform([0.,0.,0.01], [20.,20.,0.5], size=(nwalkers,3))
        sampler = emcee.EnsembleSampler(nwalkers,
                                        self.model.nparameters,
                                        self.model)
        pos,prob,state = sampler.run_mcmc(p0, 1000)
        #sampler.reset()
        #pos,prob,state = sampler.run_mcmc(pos, 1000)

        fig,axes = plt.subplots(1,3)
        axes[0].hist(sampler.flatchain[:,0], normed=True)
        axes[0].axvline(self.model.parameters['alpha'].truth.value, color='g')
        axes[0].set_xlim(0, 20)

        axes[1].hist(sampler.flatchain[:,1], normed=True)
        axes[1].axvline(self.model.parameters['beta'].truth.value, color='g')
        axes[1].set_xlim(0, 20)

        axes[2].hist(sampler.flatchain[:,2], normed=True)
        axes[2].axvline(self.model.parameters['sigma'].truth.value, color='g')
        axes[2].set_xlim(0, 0.5)

        # fig.savefig(os.path.join(plot_path,"jvp_fit_line.png"))

        ###

        fig,axes = plt.subplots(1,3)
        for walker in sampler.chain:
            axes[0].plot(walker[:,0], marker=None)
            axes[1].plot(walker[:,1], marker=None)
            axes[2].plot(walker[:,2], marker=None)

        axes[0].axhline(self.model.parameters['alpha'].truth.value, color='g')
        axes[0].set_ylim(0, 20)

        axes[1].axhline(self.model.parameters['beta'].truth.value, color='g')
        axes[1].set_ylim(0, 20)

        axes[2].axhline(self.model.parameters['sigma'].truth.value, color='g')
        axes[2].set_ylim(0, 0.5)

        # fig.savefig(os.path.join(plot_path,"jvp_fit_line_trace.png"))

