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

class ModelParameter(u.Quantity):

    def __new__(cls, name, value=None, truth=None, prior=None):
        """ Represents a model Parameter MCMC inference. This object is meant
            to be used with a Model object. The value can be a vector or scalar,
            and may also have units. The default value is NaN (numpy.nan) because
            Python's built-in None cannot be used with Quantity objects, but this
            is slightly wrong because NaN will operate with other numbers...
        """

        if value is None and truth is None:
            value = np.array(np.nan)*u.dimensionless_unscaled
            truth = np.array(np.nan)*u.dimensionless_unscaled

        elif value is not None and truth is None:
            if hasattr(value, "unit"):
                unit = value.unit
            else:
                unit = u.dimensionless_unscaled
            value = np.array(value)*unit
            truth = np.ones_like(value)*np.nan

        elif value is None and truth is not None:
            if hasattr(truth, "unit"):
                unit = truth.unit
            else:
                unit = u.dimensionless_unscaled
            truth = np.array(truth)*unit
            value = np.ones_like(truth)*np.nan

        elif value is not None and truth is not None:
            if hasattr(value, "unit"):
                vunit = value.unit
            else:
                vunit = u.dimensionless_unscaled

            if hasattr(truth, "unit"):
                tunit = truth.unit
            else:
                tunit = u.dimensionless_unscaled

            if not vunit.is_equivalent(tunit):
                raise u.UnitsError("Incompatible units '{}' and '{}'"
                                   .format(vunit, tunit))

            value = np.array(value)*vunit
            truth = np.array(truth)*tunit

        self = super(ModelParameter, cls).__new__(cls, value.value,
                                                  unit=value.unit)

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

        return self

    def copy(self):
        """ Return a copy of this `ModelParameter` instance """
        return ModelParameter(name=self.name, value=self.value*self.unit,
                              prior=self._prior, truth=self.truth)

    def __deepcopy__(self):
        """ Return a copy of this `ModelParameter` instance """
        return self.copy()

    def __repr__(self):
        return "<ModelParameter '{}'>".format(self.name)

    def __str__(self):
        return self.name

    # -------------------------------------------------------------------------- #
    #   this stuff is so this object is picklable / can be used
    #       with multiprocessing and MPI.
    def __reduce__(self):
        # patch to pickle ModelParameter objects (ndarray subclasses),
        # see http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html

        object_state = list(super(ModelParameter, self).__reduce__())
        object_state[2] = (object_state[2], self.__dict__)
        return tuple(object_state)

    def __setstate__(self, state):
        # patch to unpickle ModelParameter objects (ndarray subclasses),
        # see http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html

        nd_state, own_state = state
        super(ModelParameter, self).__setstate__(nd_state)
        self.__dict__.update(own_state)
    # -------------------------------------------------------------------------- #