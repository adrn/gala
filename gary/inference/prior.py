# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["BasePrior", "UniformPrior", "LogarithmicPrior", "NormalPrior"]

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
        """ Uniform distribution. Returns 0 if value is outside of the
            ND hyperrectangle defined by the (vectors) a, b. Returns
            the properly normalized constant prod(1/(b-a)) otherwise.

            Parameters
            ----------
            a : numeric, quantity_like, array_like
                Lower bound.
            b : numeric, quantity_like, array_like
                Lower bound.
        """

        self.a = np.atleast_1d(a)
        self.b = np.atleast_1d(b)

        if self.a.shape != self.b.shape:
            raise ValueError("Shape of 'a' must match shape of 'b'.")

        if self.a.ndim > 1:
            raise ValueError("Only one dimensional distributions supported.")

    def pdf(self, x):
        x = np.atleast_1d(x)
        p = np.zeros_like(x)

        ix = (x < self.a) | (x > self.b)
        p[ix] = 0.
        p[~ix] = (1 / (self.b - self.a))[~ix]

        return np.squeeze(p)

    def logpdf(self, x):
        x = np.atleast_1d(x)
        p = np.zeros_like(x)

        ix = (x < self.a) | (x > self.b)
        p[ix] = -np.inf
        p[~ix] = (-np.log(self.b - self.a))[~ix]

        return np.squeeze(p)

    def sample(self, n=None):
        """
        Sample from this prior. The returned array axis=0 is the
        sample axis.

        Parameters
        ----------
        n : int (optional)
            Number of samples to draw
        """
        if n is not None and self.a.size > 1:
            return np.random.uniform(self.a, self.b, size=(n,self.a.size))
        elif n is not None and self.a.size == 1:
            return np.random.uniform(self.a, self.b, size=(n,))
        else:
            return np.random.uniform(self.a, self.b)

    def __str__(self):
        return "<Uniform a={}, b={}>".format(self.a, self.b)

class LogarithmicPrior(BasePrior):

    def __init__(self, a, b):
        """ Logarithmic (scale-invariant) prior. Returns 0 if value is
            outside of the range defined by a < value < b. Otherwise,
            returns ln(b/a)/value.

            Parameters
            ----------
            a : numeric, quantity_like, array_like
                Lower bound.
            b : numeric, quantity_like, array_like
                Lower bound.
        """
        self.a = np.atleast_1d(a)
        self.b = np.atleast_1d(b)

        if self.a.shape != self.b.shape:
            raise ValueError("Shape of 'a' must match shape of 'b'.")

        if self.a.ndim > 1:
            raise ValueError("Only one dimensional distributions supported.")

    def pdf(self, x):
        x = np.atleast_1d(x)
        p = np.zeros_like(x)

        ix = (x < self.a) | (x > self.b)
        p[ix] = 0.
        p[~ix] = (1 / np.log(self.b / self.a))[~ix]

        return np.squeeze(p)

    def logpdf(self, x):
        x = np.atleast_1d(x)
        p = np.zeros_like(x)

        ix = (x < self.a) | (x > self.b)
        p[ix] = -np.inf
        p[~ix] = (np.log(1. / np.log(self.b/self.a)))[~ix]

        return np.squeeze(p)

    def sample(self, n=None):
        """
        Sample from this prior. The returned array axis=0 is the
        sample axis.

        Parameters
        ----------
        n : int (optional)
            Number of samples to draw
        """
        if n is not None and self.a.size > 1:
            return np.exp(np.random.uniform(self.a, self.b, size=(n,self.a.size)))
        elif n is not None and self.a.size == 1:
            return np.exp(np.random.uniform(self.a, self.b, size=(n,)))
        else:
            return np.exp(np.random.uniform(self.a, self.b))

    def __str__(self):
        return "<Logarithmic a={}, b={}>".format(self.a, self.b)

class NormalPrior(BasePrior):

    def __init__(self, mean, stddev):
        """ Normal (Gaussian) prior.

            Parameters
            ----------
            mean : numeric, quantity_like, array_like
                Mean of the distribution.
            stddev : numeric, quantity_like, array_like
                Standard of deviation / square root of variance.
        """

        self.mean = np.atleast_1d(mean)
        self.stddev = np.atleast_1d(stddev)
        self._norm = -0.5*np.log(2*np.pi) - np.log(self.stddev)

    def __str__(self):
        return "<Normal μ={}, σ={}>".format(self.mean, self.stddev)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        x = np.atleast_1d(x)
        xx = self.mean - x
        return self._norm - 0.5*(xx / self.stddev)**2

    def sample(self, n=None):
        """ Sample from this prior. The returned array axis=0 is the
            sample axis.

            Parameters
            ----------
            n : int (optional)
                Number of samples to draw
        """
        if n is not None and self.mean.size > 1:
            return np.random.normal(self.mean, self.stddev, size=(n,self.mean.size))
        elif n is not None and self.mean.size == 1:
            return np.random.normal(self.mean, self.stddev, size=(n,))
        else:
            return np.random.normal(self.mean, self.stddev)
