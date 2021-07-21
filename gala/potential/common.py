# Standard library
import inspect

# Third-party
import astropy.units as u
from astropy.utils import isiterable
import numpy as np

# Project
from ..util import atleast_2d
from ..units import UnitSystem, DimensionlessUnitSystem


class PotentialParameter:
    """A class for defining parameters needed by the potential classes

    Parameters
    ----------
    name : str
        The name of the parameter. For example, "m" for mass.
    physical_type : str (optional)
        The physical type (as defined by `astropy.units`) of the expected
        physical units that this parameter is in. For example, "mass" for a mass
        parameter.
    default : numeric, str, array (optional)
        The default value of the parameter.
    equivalencies : `astropy.units.equivalencies.Equivalency` (optional)
        Any equivalencies required for the parameter.
    """

    def __init__(self, name, physical_type="dimensionless", default=None,
                 equivalencies=None):
        # TODO: could add a "shape" argument?
        # TODO: need better sanitization and validation here

        self.name = str(name)
        self.physical_type = str(physical_type)
        self.default = default
        self.equivalencies = equivalencies

    def __repr__(self):
        return f"<PotentialParameter: {self.name} [{self.physical_type}]>"


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
            if k == 'self' or param.kind == param.VAR_POSITIONAL:
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

    def _parse_parameter_values(self, *args, **kwargs):
        expected_parameter_keys = list(self._parameters.keys())

        if len(args) > len(expected_parameter_keys):
            raise ValueError(
                "Too many positional arguments passed in to "
                f"{self.__class__.__name__}: Potential and Frame classes only "
                "accept parameters as positional arguments, all other "
                "arguments (e.g., units) must now be passed in as keyword "
                "argument.")

        parameter_values = dict()

        # Get any parameters passed as positional arguments
        i = 0

        if args:
            for i in range(len(args)):
                parameter_values[expected_parameter_keys[i]] = args[i]
            i += 1

        # Get parameters passed in as keyword arguments:
        for k in expected_parameter_keys[i:]:
            val = kwargs.pop(k, self._parameters[k].default)
            parameter_values[k] = val

        if len(kwargs):
            raise ValueError(f"{self.__class__} received unexpected keyword "
                             f"argument(s): {list(kwargs.keys())}")

        return parameter_values

    @classmethod
    def _prepare_parameters(cls, parameters, units):

        pars = dict()
        for k, v in parameters.items():
            expected_ptype = cls._parameters[k].physical_type
            expected_unit = units[expected_ptype]
            equiv = cls._parameters[k].equivalencies

            if hasattr(v, 'unit'):
                if (not isinstance(units, DimensionlessUnitSystem) and
                        not v.unit.is_equivalent(expected_unit, equiv)):
                    msg = (f"Parameter {k} has physical type "
                           f"'{v.unit.physical_type}', but we expected a "
                           f"physical type '{expected_ptype}'")
                    if equiv is not None:
                        msg = (msg +
                               f" or something equivalent via the {equiv} "
                               "equivalency.")

                    raise ValueError(msg)

                # NOTE: this can lead to some comparison issues in __eq__, which
                # tests for strong equality between parameter values. Here, the
                # .to() could cause small rounding issues in comparisons
                if v.unit.physical_type != expected_ptype:
                    v = v.to(expected_unit, equiv)

            elif expected_ptype is not None:
                # this is false for empty ptype: treat empty string as u.one
                # (i.e. this goes to the else clause)

                # TODO: remove when fix potentials that ask for scale velocity!
                if expected_ptype == 'speed':
                    v = v * units['length'] / units['time']
                else:
                    v = v * units[expected_ptype]

            else:
                v = v * u.one

            pars[k] = v.decompose(units)

        return pars

    def _remove_units_prepare_shape(self, x):
        from gala.dynamics import PhaseSpacePosition

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
        par_bool = [
            (k1 == k2) and np.all(self.parameters[k1] == other.parameters[k2])
            for k1, k2 in zip(self.parameters.keys(), other.parameters.keys())]
        return np.all(par_bool) and (str(self) == str(other)) and (self.units == other.units)

    # String representations:
    def __repr__(self):
        pars = []

        keys = self.parameters.keys()
        for k in keys:
            v = self.parameters[k].value
            post = ""

            if hasattr(v, 'unit'):
                post = f" {v.unit}"
                v = v.value

            if isinstance(v, float):
                if v == 0:
                    par = f"{v:.0f}"
                elif np.log10(v) < -2 or np.log10(v) > 5:
                    par = f"{v:.2e}"
                else:
                    par = f"{v:.2f}"

            elif isinstance(v, int) and np.log10(v) > 5:
                par = f"{v:.2e}"

            else:
                par = str(v)

            pars.append(f"{k}={par}{post}")

        par_str = ", ".join(pars)

        if isinstance(self.units, DimensionlessUnitSystem):
            return f"<{self.__class__.__name__}: {par_str} (dimensionless)>"
        else:
            core_units_str = ",".join(map(str, self.units._core_units))
            return f"<{self.__class__.__name__}: {par_str} ({core_units_str})>"

    def __str__(self):
        return self.__class__.__name__
