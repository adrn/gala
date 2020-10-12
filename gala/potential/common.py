# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
from astropy.utils import isiterable
import numpy as np

# Project
from ..dynamics import PhaseSpacePosition
from ..util import atleast_2d
from ..units import UnitSystem, DimensionlessUnitSystem


class CommonBase(object):

    def _validate_units(self, units):

        # make sure the units specified are a UnitSystem instance
        if units is not None and not isinstance(units, UnitSystem):
            units = UnitSystem(*units)

        elif units is None:
            units = DimensionlessUnitSystem()

        return units

    @classmethod
    def _prepare_parameters(cls, parameters, units):

        pars = OrderedDict()
        for k, v in parameters.items():
            expected_ptype = cls._parameters[k].physical_type
            if hasattr(v, 'unit'):
                if v.unit.physical_type != expected_ptype:
                    raise ValueError(
                        f"Parameter {k} has physical type "
                        f"'{v.unit.physical_type}', but we expected a physical "
                        f"type '{expected_ptype}'")
                pars[k] = v.decompose(units)

            elif expected_ptype is not None:
                # this is false for empty ptype: treat empty string as u.one
                # (i.e. this goes to the else clause)

                # TODO: remove when fix potentials that ask for scale velocity!
                if expected_ptype == 'speed':
                    pars[k] = v * units['length'] / units['time']
                else:
                    pars[k] = v * units[expected_ptype]

            else:
                pars[k] = v * u.one

        return pars

    def _remove_units_prepare_shape(self, x):
        if hasattr(x, 'unit'):
            x = x.decompose(self.units).value

        elif isinstance(x, PhaseSpacePosition):
            x = x.w(self.units)

        x = atleast_2d(x, insert_axis=1).astype(np.float64)
        return x

    def _get_c_valid_arr(self, x):
        """
        Warning! Interpretation of axes is different for C code.
        """
        orig_shape = x.shape
        x = np.ascontiguousarray(x.reshape(orig_shape[0], -1).T)
        return orig_shape, x

    def _validate_prepare_time(self, t, pos_c):
        """
        Make sure that t is a 1D array and compatible with the C position array.
        """
        if hasattr(t, 'unit'):
            t = t.decompose(self.units).value

        if not isiterable(t):
            t = np.atleast_1d(t)

        t = np.ascontiguousarray(t.ravel())

        if len(t) > 1:
            if len(t) != pos_c.shape[0]:
                raise ValueError("If passing in an array of times, it must have a shape "
                                 "compatible with the input position(s).")

        return t

    # For comparison operations
    def __eq__(self, other):
        if other is None or not hasattr(other, 'parameters'):
            return False

        # the funkiness in the below is in case there are array parameters:
        par_bool = [(k1==k2) and np.all(self.parameters[k1] == other.parameters[k2])
                    for k1,k2 in zip(self.parameters.keys(), other.parameters.keys())]
        return np.all(par_bool) and (str(self) == str(other)) and (self.units == other.units)
