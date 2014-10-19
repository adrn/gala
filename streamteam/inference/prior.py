# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["BasePrior", "UniformPrior", "Normal1DPrior"]

logger = logging.getLogger(__name__)

class BasePrior(object):

    def pdf(self, value):
        return 1.

    def logpdf(self, value):
        return 0.

    def sample(self, n=None):
        """
        Sample from this prior. The returned array axis=0 is the
        sample axis.

        Parameters
        ----------
        n : int (optional)
            Number of samples to draw
        """
        raise ValueError("Cannot sample from a BasePrior object.")

    def __str__(self):
        return "<BasePrior>"

    def __repr__(self):
        return self.__str__()

class UniformPrior(BasePrior):

    def __init__(self, a, b):
        """ Uniform distribution. Returns 0 if value is
            outside of the range defined by a < value < b.
            Returns 1/(b-a) otherwise.

            Parameters
            ----------
            a : numeric, quantity_like, array_like
                Lower bound.
            b : numeric, quantity_like, array_like
                Lower bound.
        """
        self.a = np.array(a)
        self.b = np.array(b)

    def pdf(self, value):
        value = np.array(value)
        p = np.array(1 / (self.b - self.a))
        p[(value < self.a) | (value > self.b)] = 0.
        return p

    def logpdf(self, value):
        value = np.array(value)
        p = np.array(-np.log(self.b - self.a))
        p[(value < self.a) | (value > self.b)] = -np.inf
        return p

    def sample(self, n=None):
        """
        Sample from this prior. The returned array axis=0 is the
        sample axis.

        Parameters
        ----------
        n : int (optional)
            Number of samples to draw
        """
        if n is not None:
            return np.random.uniform(self.a, self.b, size=(n,) + self.a.shape)
        else:
            return np.random.uniform(self.a, self.b)

    def __str__(self):
        return "<Uniform a={}, b={}>".format(self.a, self.b)

class Normal1DPrior(BasePrior):

    def __init__(self, mean, stddev):
        self.mean = np.array(mean)
        self.stddev = np.array(stddev)
        self._norm = -0.5*np.log(2*np.pi) - np.log(self.stddev)

    def __str__(self):
        return "<Normal1D μ={}, σ={}>".format(self.mean, self.stddev)

    def eval(self, value):
        X = self.mean - value
        return self._norm - 0.5*(X / self.stddev)**2

    def sample(self, n=None):
        """ Sample from this prior. The returned array axis=0 is the
            sample axis.

            Parameters
            ----------
            n : int (optional)
                Number of samples to draw
        """
        if n is not None:
            return np.random.normal(self.mean, self.stddev,
                                    size=(n,) + self.mean.shape)
        else:
            return np.random.normal(self.mean, self.stddev)

