# Standard library
import inspect

# Third-party
import astropy.units as u
from astropy.utils import isiterable
import numpy as np

# Project
from ..dynamics import PhaseSpacePosition
from ..util import atleast_2d
from ..units import UnitSystem, DimensionlessUnitSystem


class PotentialParameter:

    def __init__(self, name, physical_type, default=None, repr_latex=None):

        if repr_latex is None:
            repr_latex = name

        self.name = str(name)
        self.physical_type = str(physical_type)
        self.repr_latex = repr_latex
        self.default = default


class CommonBase:

    def __init_subclass__(cls, GSL_only=False, **kwargs):

        # Read the default call signature for the init
        sig = inspect.signature(cls.__init__)

        # Collect all potential parameters defined on the class:
        cls._parameters = dict()
        sig_parameters = []

        # Also allow passing parameters in to subclassing:
        subcls_params = kwargs.pop('parameters', {})
        subcls_params.update(cls.__dict__)

        for k, v in subcls_params.items():
            if not isinstance(v, PotentialParameter):
                continue

            cls._parameters[k] = v

            if v.default is None:
                default = inspect.Parameter.empty
            else:
                default = v.default

            sig_parameters.append(inspect.Parameter(
                k, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default))

        for k, param in sig.parameters.items():
            if k == 'self':
                continue
            sig_parameters.append(param)
        sig_parameters = sorted(sig_parameters, key=lambda x: int(x.kind))

        # Define a new init signature based on the potential parameters:
        newsig = sig.replace(parameters=tuple(sig_parameters))
        cls.__signature__ = newsig

        super().__init_subclass__(**kwargs)

        cls._GSL_only = GSL_only

    def _validate_units(self, units):

        # make sure the units specified are a UnitSystem instance
        if units is not None and not isinstance(units, UnitSystem):
            units = UnitSystem(*units)

        elif units is None:
            units = DimensionlessUnitSystem()

        return units

    @classmethod
    def _prepare_parameters(cls, parameters, units):

        pars = dict()
        for k, v in parameters.items():
            expected_ptype = cls._parameters[k].physical_type
            if hasattr(v, 'unit'):
                # if v.unit.physical_type != expected_ptype:
                #     raise ValueError(
                #         f"Parameter {k} has physical type "
                #         f"'{v.unit.physical_type}', but we expected a physical "
                #         f"type '{expected_ptype}'")
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
