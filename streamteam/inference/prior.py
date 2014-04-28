# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["LogPrior", "LogUniformPrior", "LogNormalPrior"]

logger = logging.getLogger(__name__)

class LogPrior(object):

    def __call__(self, value):
        return 0.

    def sample(self, size=None):
        return np.nan

class LogUniformPrior(LogPrior):

    def __str__(self):
        return "Uniform({}, {})".format(self.a, self.b)

    def __repr__(self):
        return self.__str__()

    def __call__(self, value):
        if np.any((value < self.a) | (value > self.b)):
            return -np.inf
        return 0.0

    def __init__(self, a, b):
        """ Log of a uniform distribution. Returns 0 if value is
            outside of the range defined by a < value < b.

            Parameters
            ----------
            a : numeric, quantity_like, array_like
                Lower bound.
            b : numeric, quantity_like, array_like
                Lower bound.
        """
        self.a = np.atleast_1d(a)
        self.b = np.atleast_1d(b)
        self.shape = self.a.shape

        if self.a.shape != self.b.shape:
            raise ValueError("a and b must match in shape!")

    def sample(self, size=1):
        return np.random.uniform(self.a, self.b, size=(size,)+self.shape)

class LogNormalPrior(LogPrior):

    def __call__(self, value):
        value = np.atleast_1d(value)
        if value.ndim <= 1:
            value = np.atleast_2d(value).T.copy()
        else:
            value = np.atleast_2d(value)

        X = self.mean - value
        arg = np.array([np.dot(XX, np.dot(IC,XX)) for XX,IC in zip(X,self.icov)])
        return self._norm - 0.5*arg

    def __init__(self, mean, stddev=None, cov=None):
        """ Log of a normal distribution. Need to specify either
            stddev, an array of standard deviations, or cov, an
            array of covariance matrices.

            Shape guide:
                (n,)  -> n 1-D Gaussians
                (n,m) -> n m-D Gaussians

            Parameters
            ----------
            mean : array_like
            stddev : array_like
            cov : array_like
        """
        mean = np.atleast_1d(mean)
        if mean.ndim <= 1:
            self.mean = np.atleast_2d(mean).T.copy()
        else:
            self.mean = np.atleast_2d(mean)

        self.shape = self.mean.shape
        self.ndim = self.shape[-1]

        if stddev is not None:
            if mean.ndim <= 1:
                stddev = np.atleast_2d(stddev).T.copy()
            else:
                stddev = np.atleast_2d(stddev)

            if stddev.shape != self.shape:
                raise ValueError("stddev has wrong shape {}, should be {}"
                                 .format(stddev.shape,self.shape))

            cov = np.array([np.diag(s**2) for s in stddev])

        elif cov is not None:
            if cov.ndim < 2:
                raise ValueError("Invalid covariance matrix - must have 2 dims.")
            elif cov.ndim == 2:
                cov = cov.reshape((1,)+cov.shape)

        else:
            raise ValueError("You must specify stddev or cov.")

        self.stddev = stddev
        self.cov = cov
        self.icov = np.linalg.inv(cov)

        dets = np.linalg.slogdet(self.cov)[1]
        self._norm = -0.5*self.ndim*np.log(2*np.pi) - 0.5*dets

    def sample(self, size=1):
        s = np.array([np.random.multivariate_normal(m,C,size=size)
                        for m,C in zip(self.mean, self.cov)])

        return np.rollaxis(s, 1)