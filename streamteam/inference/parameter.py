# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import astropy.units as u
from astropy.utils import isiterable

# Project
from .. import usys
from .prior import LogPrior

__all__ = ["ModelParameter"]

logger = logging.getLogger(__name__)

class ModelParameter(u.Quantity):

    def __new__(cls, name, value=np.nan, prior=None, truth=None):
        """ Represents a model parameter to be used in inference.
            `value` must be set so the object knows the shape of the
            parameter, e.g., whether it is a vector or scalar.

            Parameters
            ----------
            name : str
                Parameter name.
            value : quantity_like
            prior : LogPrior
            truth : quantity_like
        """

        value = np.atleast_1d(value)
        if hasattr(value, "unit"):
            _value = value.decompose(usys)
        else:
            _value = value*u.dimensionless_unscaled

        self = super(ModelParameter, cls).__new__(cls, value)

        # make sure input prior is a Prior, or a list of Prior objects
        if prior is None:
            prior = LogPrior()

        if truth is None:
            truth = np.zeros_like(_value)*np.nan

        if not isinstance(prior, LogPrior):
            raise TypeError("prior must be a LogPrior subclass, not {}."
                            .format(type(prior)))

        self.prior = prior
        self.truth = truth
        self.name = str(name)

        return self

    def copy(self):
        """ Return a copy of this `ModelParameter` instance """
        return ModelParameter(name=self.name, value=self.value*self.unit,
                              prior=self.prior, truth=self.truth)

    def __deepcopy__(self):
        """ Return a copy of this `ModelParameter` instance """
        return self.copy()

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

    def __repr__(self):
        extra = ""
        if self.truth is not None:
            extra = " truth={}".format(self.truth)

        return "<ModelParameter '{}'{}>".format(self.name, extra)

    def __str__(self):
        return self.name