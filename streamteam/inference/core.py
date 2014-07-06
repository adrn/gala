# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
from collections import OrderedDict

# Third-party
import numpy as np
import astropy.units as u
from astropy import log as logger

# Project
from .parameter import ModelParameter
from .prior import *

__all__ = ["EmceeModel"]

def walk_dict(d):
    group = None
    for name,param in d.items():
        if hasattr(param,"items"):
            group = name
            for grp,name,param in walk_dict(param):
                if param.frozen: # skip frozen parameters
                    continue
                else:
                    yield group, name, param
        else:
            yield group, name, param

class EmceeModel(object):

    def __init__(self, ln_likelihood, ln_prior=None, args=()):
        """ """

        self.parameters = OrderedDict()
        self.nparameters = 0

        if ln_prior is not None:
            self.ln_prior = ln_prior

        self.ln_likelihood = ln_likelihood
        self.args = args

        if not hasattr(self.ln_likelihood, "__call__"):
            raise TypeError("ln_likelihood must be callable.")
        elif not hasattr(self.ln_prior, "__call__"):
            raise TypeError("ln_prior must be callable.")

    def add_parameter(self, param, group=None):
        """ Add a parameter to the model.

            Parameters
            ----------
            param : ModelParameter
                The parameter instance.
        """

        if not isinstance(param, ModelParameter):
            raise TypeError("Invalid parameter type '{}'".format(type(param)))

        if group is not None and group not in self.parameters.keys():
            self.parameters[group] = OrderedDict()
            self.parameters[group][param.name] = param.copy()
        else:
            self.parameters[param.name] = param.copy()

        self.nparameters += param.size

    def _walk(self):
        """ Walk through a dictionary tree with maximum depth=2 """
        for tup in walk_dict(self.parameters):
            yield tup

    def ln_prior(self, parameters, value_dict, *args):
        """ Default prior -- if none specified, evaluates the priors
            over each parameter at the specified dictionary of values
        """
        ln_prior = 0.
        for group_name,param_name,param in self._walk():
            if group_name is None:
                v = value_dict[param_name]
            else:
                v = value_dict[group_name][param_name]
            ln_prior += param.prior(v)

        return ln_prior

    @property
    def truth_vector(self):
        """ Returns an array of the true values of all parameters in the model """

        true_p = np.array([])
        for group_name,param_name,param in self._walk():
            true_p = np.append(true_p, np.ravel(param.truth))

        return true_p

    def devectorize(self, p):
        """ Turns a vector of parameter values, e.g. from MCMC, and turns it into
            a dictionary of parameter values.

            Parameters
            ----------
            p : array_like
                The vector of model parameter values.
        """
        d = OrderedDict()

        ix1 = 0
        for group_name,param_name,param in self._walk():
            val = p[ix1:ix1+param.size]
            if group_name is None:
                d[param_name] = val
            else:
                d[group_name][param_name] = val

            ix1 += param.size

        return d

    def vectorize(self, value_dict):
        """ Turn a parameter dictionary into a parameter vector

            Parameters
            ----------
            param_dict : OrderedDict
        """

        vec = np.array([])
        for group_name,param_name,param in self._walk():
            if group_name is None:
                p = np.ravel(value_dict[param_name])
            else:
                p = np.ravel(value_dict[group_name][param_name])
            vec = np.append(vec, p)

        return vec

    def ln_posterior(self, values):
        ln_prior = self.ln_prior(self.parameters, values, *self.args)

        # short-circuit if any prior value is -infinity
        if np.any(np.isinf(ln_prior)):
            return -np.inf

        ln_like = self.ln_likelihood(self.parameters, values, *self.args)

        # short-circuit if any likelihood value is -infinity
        if np.any(np.isinf(ln_like)):
            return -np.inf

        # sum over each star
        ln_prior = np.sum(ln_prior)
        ln_like = np.sum(ln_like)

        if np.isnan(ln_like):
            raise ValueError("Likelihood returned NaN value.")
        elif np.isnan(ln_prior):
            raise ValueError("Prior returned NaN value.")

        return ln_like + ln_prior

    def __call__(self, p):
        value_dict = self.devectorize(p)
        return self.ln_posterior(value_dict)

    def sample_priors(self, size=1):
        """ Draw samples from the priors over the model parameters.

            Parameters
            ----------
            size : int
                Number of samples to draw.
            start_truth : bool (optional)
                Sample centered on true values.
        """

        p0 = np.zeros((size, self.nparameters))
        ix1 = 0
        for group_name,param_name,param in self._walk():
            p0[:,ix1:ix1+param.size] = param.prior.sample(size=size)
            ix1 += param.size

        return p0
