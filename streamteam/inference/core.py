# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
from collections import OrderedDict

# Third-party
import numpy as np
import astropy.units as u

# Project
from .parameter import ModelParameter
from .prior import *

__all__ = ["EmceeModel"]

logger = logging.getLogger(__name__)

class EmceeModel(object):

    def __init__(self, ln_likelihood=None, ln_likelihood_args=()):
        """ """

        self.parameters = OrderedDict()
        self.nparameters = 0

        if ln_likelihood is None:
            ln_likelihood = lambda *args,**kwargs: 0.

        self.ln_likelihood = ln_likelihood
        self.ln_likelihood_args = ln_likelihood_args

    def add_parameter(self, parameter, parameter_group=None):
        """ Add a parameter to the model.

            Parameters
            ----------
            parameter : ModelParameter
                The parameter instance.
            parameter_group : str (optional)
                Name of the parameter group to add this parameter to.
        """

        if not isinstance(parameter, ModelParameter):
            raise TypeError("Invalid parameter type '{}'".format(type(parameter)))

        if parameter_group is None:
            parameter_group = "main"

        if parameter_group not in self.parameters.keys():
            self.parameters[parameter_group] = OrderedDict()

        self.parameters[parameter_group][parameter.name] = parameter.copy()
        self.nparameters += parameter.size

    def _walk(self, container):
        for group_name,group in container.items():
            for param_name,param in group.items():
                yield group_name, param_name, param

    @property
    def truths(self):
        """ Returns an array of the true values of all parameters in the model """

        true_p = np.array([])
        for group_name,param_name,param in self._walk(self.parameters):
            true_p = np.append(true_p, np.ravel(param.truth))

        return true_p

    def vector_to_parameters(self, p):
        """ Turns a vector of parameter values, e.g. from MCMC, and turns it into
            a dictionary of parameters.

            Parameters
            ----------
            p : array_like
                The vector of model parameter values.
        """
        d = OrderedDict()

        ix1 = 0
        for group_name,param_name,param in self._walk(self.parameters):
            if group_name not in d.keys():
                d[group_name] = OrderedDict()

            par = ModelParameter(name=param_name, value=p[ix1:ix1+param.size],
                                 truth=param.truth, prior=param.prior)
            d[group_name][param_name] = par# p[ix1:ix1+param.size]
            ix1 += param.size

        return d

    def parameters_to_vector(self, parameters):
        """ Turn a parameter dictionary into a parameter vector

            Parameters
            ----------
            param_dict : OrderedDict
        """

        vec = np.array([])
        for group_name,param_name,param in self._walk(parameters):
            p = np.ravel(parameters[group_name][param_name].value)
            vec = np.append(vec, p)

        return vec

    def ln_prior(self, parameters):
        ln_prior = 0.
        for group_name,param_name,param in self._walk(parameters):
            lp = param.prior(param.value)
            ln_prior += lp

        return ln_prior

    def ln_posterior(self, parameters):
        ln_prior = self.ln_prior(parameters)

        # short-circuit if any prior value is -infinity
        if np.any(np.isinf(ln_prior)):
            return -np.inf

        ln_like = self.ln_likelihood(parameters, *self.ln_likelihood_args)

        if np.any(np.isnan(ln_like)) or np.any(np.isnan(ln_prior)):
            return -np.inf

        return np.sum(ln_like) + np.sum(ln_prior)

    def __call__(self, p):
        parameters = self.vector_to_parameters(p)
        return self.ln_posterior(parameters)

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
        for group_name,param_name,param in self._walk(self.parameters):
            p0[:,ix1:ix1+param.size] = param.prior.sample(size=size)
            ix1 += param.size

        return p0
