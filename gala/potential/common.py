# coding: utf-8

from __future__ import division, print_function

# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np

# Project
from ...util import atleast_2d
from ...units import UnitSystem, DimensionlessUnitSystem

class PotentialCommmonBase(object):

    def _validate_units(self, units):

        # make sure the units specified are a UnitSystem instance
        if units is not None and not isinstance(units, UnitSystem):
            units = UnitSystem(*units)

        elif units is None:
            units = DimensionlessUnitSystem()

        return units

    def _prepare_parameters(self, parameters, units):
        pars = OrderedDict()
        for k,v in parameters.items():
            if hasattr(v, 'unit'):
                pars[k] = v.decompose(units)
            else:
                pars[k] = v*u.one
        return pars

    def _remove_units_prepare_shape(self, x):
        if hasattr(x, 'unit'):
            x = x.decompose(self.units).value
        x = atleast_2d(x, insert_axis=1).astype(np.float64)
        return np.ascontiguousarray(x)

    def _get_c_valid_arr(self, x):
        orig_shape = x.shape
        x = np.ascontiguousarray(x.reshape(orig_shape[0], -1)) # ravel to (ndim, all_others)
        return orig_shape, x
