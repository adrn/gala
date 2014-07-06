# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u

__all__ = ["LogPrior", "LogUniformPrior", "LogNormal1DPrior"]

logger = logging.getLogger(__name__)

class LogPrior(object):

    def __call__(self, value):
        return self.eval(value)

    def sample(self, n=None):
        """ Sample from this prior.

            Parameters
            ----------
            n : int (optional)
                Number of samples to draw
        """
        raise ValueError("Cannot sample from a LogPrior object.")

    def __str__(self):
        return "<LogPrior>"

    def __repr__(self):
        return self.__str__()

class LogUniformPrior(LogPrior):

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

        if self.a.shape != self.b.shape:
            raise ValueError("a and b must match in shape!")

    def __str__(self):
        return "<Uniform a={}, b={}>".format(self.a, self.b)

    def eval(self, value):
        p = np.log(1 / (self.b - self.a))
        p[(value < self.a) | (value > self.b)] = -np.inf
        return p

    def sample(self, n=None):
        """ Sample from this prior.

            Parameters
            ----------
            n : int (optional)
                Number of samples to draw
        """
        if n is not None:
            return np.random.uniform(self.a, self.b, size=(n,) + self.a.shape)
        else:
            return np.random.uniform(self.a, self.b)

class LogNormal1DPrior(LogPrior):

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
        """ Sample from this prior.

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

# class LogNormalPrior(LogPrior):

#     def __call__(self, value):
#         value = np.atleast_1d(value)
#         if value.ndim <= 1:
#             value = np.atleast_2d(value).T.copy()
#         else:
#             value = np.atleast_2d(value)

#         X = self.mean - value
#         arg = np.array([np.dot(XX, np.dot(IC,XX)) for XX,IC in zip(X,self.icov)])
#         return self._norm - 0.5*arg

#     def __init__(self, mean, stddev=None, cov=None):
#         """ Log of a normal distribution. Need to specify either
#             stddev, an array of standard deviations, or cov, an
#             array of covariance matrices.

#             Shape guide:
#                 (n,)  -> n 1-D Gaussians
#                 (n,m) -> n m-D Gaussians

#             Parameters
#             ----------
#             mean : array_like
#             stddev : array_like
#             cov : array_like
#         """
#         mean = np.atleast_1d(mean)
#         if mean.ndim <= 1:
#             self.mean = np.atleast_2d(mean).T.copy()
#         else:
#             self.mean = np.atleast_2d(mean)

#         self.shape = self.mean.shape
#         self.ndim = self.shape[-1]

#         if stddev is not None:
#             if mean.ndim <= 1:
#                 stddev = np.atleast_2d(stddev).T.copy()
#             else:
#                 stddev = np.atleast_2d(stddev)

#             if stddev.shape != self.shape:
#                 raise ValueError("stddev has wrong shape {}, should be {}"
#                                  .format(stddev.shape,self.shape))

#             cov = np.array([np.diag(s**2) for s in stddev])

#         elif cov is not None:
#             if cov.ndim < 2:
#                 raise ValueError("Invalid covariance matrix - must have 2 dims.")
#             elif cov.ndim == 2:
#                 cov = cov.reshape((1,)+cov.shape)

#         else:
#             raise ValueError("You must specify stddev or cov.")

#         self.stddev = stddev
#         self.cov = cov
#         self.icov = np.linalg.inv(cov)

#         dets = np.linalg.slogdet(self.cov)[1]
#         self._norm = -0.5*self.ndim*np.log(2*np.pi) - 0.5*dets

#     def sample(self, size=1):
#         s = np.array([np.random.multivariate_normal(m,C,size=size)
#                         for m,C in zip(self.mean, self.cov)])

#         return np.rollaxis(s, 1)