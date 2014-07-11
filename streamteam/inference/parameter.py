# coding: utf-8

""" Parameter classes """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u
from astropy import log as logger

# Project
from .prior import LogPrior

__all__ = ["ModelParameter"]

class ModelParameter(object):

    def __init__(self, name, truth=None, prior=None, shape=None):
        """ Represents a model Parameter MCMC inference. This object is meant
            to be used with a Model object. The value can be a vector or scalar,
            and may also have units. The default value is NaN (numpy.nan) because
            Python's built-in None cannot be used with Quantity objects, but this
            is slightly wrong because NaN will operate with other numbers...
        """

        if truth is None and shape is None: # assume scalar
            truth = np.array(np.nan)*u.dimensionless_unscaled

        elif truth is None and shape is not None: # shape specified
            truth = np.zeros(shape)*np.nan*u.dimensionless_unscaled

        elif truth is not None and shape is None: # truth specified, not shape
            if hasattr(truth, "unit"):
                unit = truth.unit
            else:
                unit = u.dimensionless_unscaled

            if unit is None:
                unit = u.dimensionless_unscaled

            truth = np.asarray(truth)*unit

        elif truth is not None and shape is not None: # truth and shape specified
            if truth.shape != shape:
                raise ValueError("Size mismatch: truth shape ({}) does not match"
                                 "specified shape ({})".format(truth.shape, shape))

        else:
            raise WTFError()

        # assign a benign prior that always evaluates to 0. if none specified
        if prior is None:
            prior = LogPrior()

        # prior must be a LogPrior object
        if not isinstance(prior, LogPrior):
            raise TypeError("prior must be a LogPrior subclass, not {}."
                            .format(type(prior)))

        self.prior = prior
        self.truth = truth
        self.name = str(name)

        # whether the parameter is variable or not - default: not frozen
        self.frozen = False

    def freeze(self, val):
        """ Freeze the parameter to the specified value """
        if str(val).lower().strip() == "truth":
            self.frozen = self.truth
        else:
            self.frozen = val

    def thaw(self):
        """ Un-freeze (thaw) the parameter. """
        self.frozen = False

    @property
    def size(self):
        return self.truth.size

    @property
    def shape(self):
        return self.truth.shape

    @property
    def __len__(self):
        return len(self.truth)

    def copy(self):
        """ Return a copy of this `ModelParameter` instance """
        p = ModelParameter(name=self.name, truth=self.truth, prior=self.prior)
        p.frozen = self.frozen
        return p

    def __deepcopy__(self):
        """ Return a copy of this `ModelParameter` instance """
        return self.copy()

    def __repr__(self):
        return "<ModelParameter '{}'>".format(self.name)

    def __str__(self):
        return self.name
