"""
Time-interpolated potential wrapper for Gala.

This module provides the TimeInterpolatedPotential class that allows interpolating
potential parameters, origin, and rotation over time using GSL splines.
"""

import copy

import numpy as np

from ....integrate.timespec import parse_time_specification
from ...common import PotentialParameter
from ..cpotential import CPotentialBase
from .cytimeinterp import TimeInterpolatedWrapper

__all__ = ["TimeInterpolatedPotential"]

_unsupported_cls = [
    "EXPPotential",
    "HenonHeilesPotential",
    "NullPotential",
    "MultipolePotential",  # TODO?
    "MN3ExponentialDiskPotential",  # TODO: need to move parameter transforms to C
    "SphericalSplinePotential",  # TODO
    "CylSplinePotential",  # TODO
]


class TimeInterpolatedPotential(CPotentialBase, GSL_only=True):
    """
    A time-interpolated wrapper for any potential class.

    This class allows any PotentialBase subclass to have time-varying parameters,
    origin, and rotation by interpolating between values specified at discrete
    time knots using GSL splines.

    Parameters
    ----------
    potential_cls : PotentialBase subclass
        The potential class to wrap.
    time_knots : array_like
        Array of time values for interpolation knots. Must be monotonically increasing.
    interpolation_method : str, optional
        Interpolation type. Any GSL interpolation type is supported:
        https://www.gnu.org/software/gsl/doc/html/interp.html
        Common options are:
        - 'linear': Linear interpolation
        - 'cspline': Cubic spline interpolation (default)
        - 'akima': Akima spline interpolation. This avoids unphysical wiggles in
           regions where the second derivative in the underlying curve is rapidly
           changing, however it does not have a continuous second derivative.
        - 'steffen': Steffen spline interpolation. This guarantees monotonicity of the
          interpolating function between the given data points. Therefore, minima and
          maxima can only occur exactly at the data points, and there can never be
          spurious oscillations between data points.
    units : UnitSystem, optional
        Unit system for the potential
    origin : array_like, optional
        Either a constant origin vector, or an array of origin vectors with shape
        (n_knots, n_dim).
    R : array_like, optional
        Either a constant rotation matrix, or an array of rotation matrices with
        shape (n_knots, n_dim, n_dim).
    **kwargs
        Potential parameters. Each parameter can be either a constant value, or an array with shape (n_knots, *parameter_shape) for a time-varying parameter.

    Examples
    --------
    Create a Kepler potential with time-varying mass:

    >>> import astropy.units as u
    >>> from gala.potential import KeplerPotential
    >>> from gala.units import galactic
    >>>
    >>> # Time knots in Myr
    >>> times = np.linspace(0, 100, 11) * u.Myr
    >>> # Mass growing linearly with time
    >>> masses = np.linspace(1e10, 2e10, 11) * u.Msun
    >>>
    >>> pot = TimeInterpolatedPotential(
    ...     KeplerPotential, times, m=masses, units=galactic
    ... )
    >>> pot.energy([1., 0, 0] * u.pc, t=0*u.Myr)
    <Quantity [-44.98502151] kpc2 / Myr2>
    >>> pot.energy([1., 0, 0] * u.pc, t=50*u.Myr)
    <Quantity [-67.47753227] kpc2 / Myr2>

    Create a potential with a time-varying rotation:

    >>> # Rotation matrices for 90 degree rotation over 1 Gyr
    >>> R_times = np.linspace(0, 1, 11) * u.Gyr
    >>> angles = np.linspace(0, np.pi / 2, 11)
    >>> Rs = np.array([R.from_rotvec([0, 0, angle]).as_matrix() for angle in angles])
    >>> pot = gp.TimeInterpolatedPotential(
    ...     gp.LongMuraliBarPotential,
    ...     R_times,
    ...     m=1e10 * u.Msun,
    ...     a=3 * u.kpc,
    ...     b=1 * u.kpc,
    ...     c=0.5 * u.kpc,
    ...     R=Rs,
    ...     units=galactic,
    ... )
    >>> pot.gradient([5., 0, 0] * u.kpc, t=0.*u.Gyr)[0, 0]
    <Quantity 0.00207787 kpc / Myr2>
    >>> pot.gradient([5., 0, 0] * u.kpc, t=0.5*u.Gyr)[0, 0]
    <Quantity 0.0015879 kpc / Myr2>
    """

    potential_cls = PotentialParameter(
        "potential_cls", physical_type=None, python_only=True, convert=None
    )
    time_knots = PotentialParameter(
        "time_knots", ndim=1, physical_type="time", python_only=True
    )
    interpolation_method = PotentialParameter(
        "interpolation_method",
        physical_type=None,
        default="cspline",
        python_only=True,
        convert=str,
    )

    def __init__(
        self,
        *args,
        units=None,
        origin=None,
        R=None,
        **kwargs,
    ):
        tmp, _ = self._parse_parameter_values(*args, strict=False, **kwargs)

        if tmp["potential_cls"].__name__ in _unsupported_cls:
            raise NotImplementedError(
                f"TimeInterpolatedPotential does not currently support "
                f"{tmp['potential_cls'].__name__}. Raise an issue on GitHub if "
                f"you would like this to be implemented:"
                "https://github.com/adrn/gala/issues"
            )

        # HACK: ._parameters exists on the class, not the instance, but this makes a
        # *copy* exist on this instance...
        self._parameters = copy.deepcopy(self._parameters)

        # Copy parameter definitions from the wrapped potential class so the base class
        # knows what parameters to expect in kwargs
        self._potential_param_names = []
        for attr_name in tmp["potential_cls"]._parameters:
            attr = getattr(tmp["potential_cls"], attr_name)
            if isinstance(attr, PotentialParameter):
                setattr(self, attr_name, attr)
                self._parameters[attr_name] = copy.copy(attr)
                self._potential_param_names.append(attr_name)

        # Validate interpolation method vs number of knots
        n_knots = len(tmp["time_knots"])
        interp_method = tmp["interpolation_method"]
        min_knots_required = {
            "linear": 2,
            "cspline": 3,
            "akima": 5,
            "steffen": 3,
        }
        if interp_method not in min_knots_required:
            raise ValueError(
                f"Interpolation method '{interp_method}' is not recognized. "
                f"Supported methods are: {list(min_knots_required.keys())}"
            )
        min_required = min_knots_required.get(interp_method)
        if n_knots < min_required:
            raise ValueError(
                f"Interpolation method '{interp_method}' requires at least "
                f"{min_required} time knots, but only {n_knots} were provided. "
                f"Either provide more time knots or use 'linear' interpolation."
            )

        # Determine dimensionality from potential class
        self.ndim = (
            tmp["potential_cls"].ndim if hasattr(tmp["potential_cls"], "ndim") else 3
        )

        # Determine which parameters have an extra ndim over expectation
        self._interp_params = []
        for param_name in self._potential_param_names:
            pp = self._parameters[param_name]
            if param_name not in kwargs:
                if pp.default is None:
                    raise ValueError(
                        f"You must specify a value for potential parameter {param_name}"
                    )
                continue

            tmp = np.asanyarray(kwargs[param_name])
            if tmp.ndim == (pp.ndim + 1):
                # Validate that the first dimension matches the number of time knots
                if tmp.shape[0] != n_knots:
                    raise ValueError(
                        f"Parameter '{param_name}' has shape {tmp.shape} but there are "
                        f"{n_knots} time knots. For time-interpolated parameters, the first "
                        f"dimension must match the number of time knots. If you intended this "
                        f"to be a constant parameter, pass a scalar value instead of a "
                        f"length-{tmp.shape[0]} array."
                    )

                self._interp_params.append(param_name)

                # increase ndim for validation
                self._parameters[param_name].ndim += 1

        # # Validate rotation matrices are orthogonal
        # for i, rot_matrix in enumerate(rotation_matrices):
        #     if not self._is_orthogonal(rot_matrix):
        #         raise ValueError(f"Rotation matrix at index {i} is not orthogonal")

        super().__init__(
            *args,
            units=units,
            origin=origin,
            R=R,
            **kwargs,
        )

        # Additional validation of input:
        if not np.all(np.diff(self.parameters["time_knots"]) > 0):
            raise ValueError(
                "time_knots must be monotonically increasing (and no duplicate times)"
            )

    def _setup_wrapper(self, **_):
        """Set up the time interpolation wrapper."""

        # This is needed because we need to pass a dummy c_instance just to get the C
        # functions for that potential.
        # TODO: there may be a better way to pass the C functions...
        potential_cls = self.parameters["potential_cls"]
        wrapped_potential = potential_cls(
            units=self.units,
            **{
                k: (
                    self.parameters[k][0]
                    if k in self._interp_params
                    else self.parameters[k]
                )
                for k in self._potential_param_names
            },
        )

        origin_arrays = (
            np.atleast_2d(self.origin)
            if self.origin is not None
            else np.zeros(self.ndim)[np.newaxis]
        )
        assert origin_arrays.ndim == 2

        if self.R is not None:
            R_arrays = self.R if self.R.ndim == 3 else self.R[np.newaxis]
        else:
            R_arrays = np.eye(3)[np.newaxis]
        assert R_arrays.ndim == 3

        # Prepare parameter arrays for the C wrapper
        # For multi-dimensional parameters that are time-interpolated,
        # reshape them from (n_knots, d1, d2, ...) to (n_knots, d1*d2*...)
        param_arrays = {}
        param_element_counts = {}  # Track how many elements each parameter has

        # Calculate how many c_only parameters exist (e.g., nmax, lmax for SCF)
        # These are prepended to c_parameters but not in the regular parameters dict
        # TODO: need to detect potential parameters that aren't array type, like
        # SphericalSplinePotential's spline_value_type
        total_regular_param_size = 0
        for k in self._potential_param_names:
            param_val = np.atleast_1d(wrapped_potential.parameters[k].value)
            total_regular_param_size += param_val.size

        n_c_only_params = len(wrapped_potential.c_parameters) - total_regular_param_size

        # Extract c_only parameters (they're constant, so just take from wrapped_potential)
        if n_c_only_params > 0:
            c_only_params = wrapped_potential.c_parameters[:n_c_only_params]
        else:
            c_only_params = np.array([])

        for k in self._potential_param_names:
            param_val = np.atleast_1d(self.parameters[k].value)

            # If this is a time-interpolated multi-dimensional parameter,
            # flatten the extra dimensions
            if k in self._interp_params and param_val.ndim > 1:
                n_knots = len(self.parameters["time_knots"])
                # Reshape from (n_knots, d1, d2, ...) to (n_knots, d1*d2*...)
                param_reshaped = param_val.reshape(n_knots, -1)
                n_elements = param_reshaped.shape[1]
                param_element_counts[k] = n_elements
                param_arrays[k] = param_reshaped.ravel()  # Flatten to 1D row-major
            # For constant parameters, flatten if multi-dimensional
            elif param_val.ndim > 1:
                param_arrays[k] = param_val.ravel()
                param_element_counts[k] = param_val.size
            else:
                param_arrays[k] = param_val
                param_element_counts[k] = 1

        self.c_instance = TimeInterpolatedWrapper(
            self.G,
            wrapped_potential.c_instance,
            self.parameters["time_knots"].value,
            self._interp_params,
            param_arrays,
            param_element_counts,
            c_only_params,
            origins=origin_arrays,
            rotation_matrices=R_arrays,
            interpolation_method=self.parameters["interpolation_method"],
        )

    @staticmethod
    def _is_orthogonal(matrix, rtol=1e-5, atol=1e-8):
        """Check if a matrix is orthogonal."""
        return np.allclose(
            matrix @ matrix.T, np.eye(matrix.shape[0]), rtol=rtol, atol=atol
        )

    def replicate(self, **kwargs):
        """Create a copy of this potential with possibly different parameters."""
        # TODO: update this

        # Extract current parameters
        new_kwargs = {}

        # Copy time-varying parameters
        for param_name, param_array in self._param_arrays.items():
            new_kwargs[param_name] = kwargs.pop(param_name, param_array)

        # Copy other parameters
        new_kwargs.update(kwargs)

        return self.__class__(
            self._potential_cls,
            self._time_knots,
            interpolation_method=self._interpolation_method,
            units=self.units,
            origin=self._origin_arrays,
            R=self._rotation_matrices,
            **new_kwargs,
        )

    def integrate_orbit(
        self,
        w0,
        Integrator=None,
        Integrator_kwargs=None,
        cython_if_possible=True,
        save_all=True,
        **time_spec,
    ):
        """
        Integrate an orbit in the current potential using the integrator class
        provided. Uses same time specification as `Integrator()` -- see
        the documentation for `gala.integrate` for more information.

        Parameters
        ----------
        w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
            Initial conditions.
        Integrator : `~gala.integrate.Integrator` (optional)
            Integrator class to use.
        Integrator_kwargs : dict (optional)
            Any extra keyword arguments to pass to the integrator class
            when initializing. Only works in non-Cython mode.
        cython_if_possible : bool (optional)
            If there is a Cython version of the integrator implemented,
            and the potential object has a C instance, using Cython
            will be *much* faster.
        save_all : bool (optional)
            Controls whether to store the phase-space position at all intermediate
            timesteps. Set to False to store only the final values (i.e. the
            phase-space position(s) at the final timestep). Default is True.
        **time_spec
            Specification of how long to integrate. See documentation
            for `~gala.integrate.parse_time_specification`.

        Returns
        -------
        orbit : `~gala.dynamics.Orbit`
        """
        if Integrator_kwargs is None:
            Integrator_kwargs = {}
        t = parse_time_specification(self.units, **time_spec)

        # ensure timesteps are within the range of time_knots
        knot_times = self.parameters["time_knots"].decompose(self.units).value
        t_min, t_max = knot_times.min(), knot_times.max()
        if np.any(t < t_min) or np.any(t > t_max):
            raise ValueError(
                "Integration times must be within the range of the Potential's interpolation range "
                f"that you defined: [{t_min}, {t_max}] {self.units['time']}, "
                f"your orbit integration range is [{min(t)}, {max(t)}] {self.units['time']}"
            )

        return super().integrate_orbit(
            w0,
            Integrator=Integrator,
            Integrator_kwargs=Integrator_kwargs,
            cython_if_possible=cython_if_possible,
            save_all=save_all,
            t=t,
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: "
            f"{self.parameters['potential_cls'].__name__} "
            f"interpolation_method='{self.parameters['interpolation_method']}')>"
        )
