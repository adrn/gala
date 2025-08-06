"""
Time-interpolated potential wrapper for Gala.

This module provides the TimeInterpolatedPotential class that allows interpolating
potential parameters, origin, and rotation over time using GSL splines.
"""

import numpy as np

from ...common import PotentialParameter
from ..cpotential import CPotentialBase
from .cytimeinterp import TimeInterpolatedWrapper

__all__ = ["TimeInterpolatedPotential"]


class TimeInterpolatedPotential(CPotentialBase, GSL_only=True):
    """
    A time-interpolated wrapper for any potential class.

    This class allows any PotentialBase subclass to have time-varying parameters,
    origin, and rotation by interpolating between values specified at discrete
    time knots using GSL splines.

    Parameters
    ----------
    potential_cls : PotentialBase subclass
        The potential class to wrap with time interpolation
    time_knots : array_like
        Array of time values for interpolation knots. Must be monotonically increasing.
    interp_kind : str, optional
        Interpolation type. Options are:
        - 'linear': Linear interpolation
        - 'cubic': Cubic spline interpolation (default)
        - 'akima': Akima spline interpolation
        - 'steffen': Steffen spline interpolation
    units : UnitSystem, optional
        Unit system for the potential
    origin : array_like or callable, optional
        Either a constant origin vector, or an array of origin vectors with shape
        (n_knots, n_dim), or a callable that takes time and returns origin
    R : array_like or callable, optional
        Either a constant rotation matrix, or an array of rotation matrices with
        shape (n_knots, n_dim, n_dim), or a callable that takes time and returns
        rotation matrix
    **kwargs
        Potential parameters. Each parameter can be either:
        - A scalar value (constant over time)
        - An array with length n_knots (time-varying, interpolated)
        - A callable that takes time and returns parameter value

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

    Create a potential with time-varying rotation:

    >>> from scipy.spatial.transform import Rotation as R
    >>>
    >>> # Rotation matrices for 90 degree rotation over time
    >>> angles = np.linspace(0, np.pi/2, 11)
    >>> rotations = np.array([R.from_rotvec([0, 0, angle]).as_matrix()
    ...                      for angle in angles])
    >>>
    >>> pot = TimeInterpolatedPotential(
    ...     KeplerPotential, times, m=1e10*u.Msun, R=rotations, units=galactic
    ... )
    """

    def __init__(
        self,
        potential_cls,
        time_knots,
        interp_kind="cubic",
        units=None,
        origin=None,
        R=None,
        **kwargs,
    ):
        # Validate inputs
        time_knots = np.asarray(time_knots)
        if time_knots.ndim != 1:
            raise ValueError("time_knots must be 1-dimensional")
        if len(time_knots) < 2:
            raise ValueError("At least 2 time knots are required")
        if not np.all(np.diff(time_knots) > 0):
            raise ValueError("time_knots must be monotonically increasing")

        n_knots = len(time_knots)

        # Store the wrapped potential class and time information
        self._potential_cls = potential_cls
        self._time_knots = time_knots
        self._interp_kind = interp_kind

        # Copy parameter definitions from the wrapped potential class
        # This is crucial so the base class knows what parameters to expect
        for attr_name in dir(potential_cls):
            attr = getattr(potential_cls, attr_name)
            if isinstance(attr, PotentialParameter):
                setattr(self, attr_name, attr)
                self._parameters[attr_name] = attr

        # Determine dimensionality from potential class
        ndim = potential_cls.ndim if hasattr(potential_cls, "ndim") else 3

        # Process parameters
        processed_params = {}
        param_arrays = {}

        for param_name, param_value in kwargs.items():
            if param_name in ["units", "origin", "R"]:
                continue  # These are handled separately

            param_value = np.asarray(param_value)

            if param_value.ndim == 0:
                # Scalar - constant parameter
                processed_params[param_name] = param_value.item()
                param_arrays[param_name] = np.full(n_knots, param_value.item())
            elif param_value.ndim == 1:
                if len(param_value) == 1:
                    # Single value - constant parameter
                    processed_params[param_name] = param_value[0]
                    param_arrays[param_name] = np.full(n_knots, param_value[0])
                elif len(param_value) == n_knots:
                    # Time-varying parameter
                    processed_params[param_name] = param_value[
                        0
                    ]  # Use first value for init
                    param_arrays[param_name] = param_value
                else:
                    raise ValueError(
                        f"Parameter {param_name} has length {len(param_value)}, "
                        f"expected 1 or {n_knots}"
                    )
            else:
                raise ValueError(f"Parameter {param_name} must be scalar or 1D array")

        # Process origin
        origin_arrays = None
        if origin is not None:
            origin = np.asarray(origin)
            if origin.ndim == 1:
                if len(origin) == ndim:
                    # Constant origin
                    origin_arrays = origin.reshape(1, -1)
                else:
                    raise ValueError(f"Origin must have length {ndim}")
            elif origin.ndim == 2:
                if origin.shape == (n_knots, ndim):
                    # Time-varying origin
                    origin_arrays = origin
                elif origin.shape == (1, ndim):
                    # Single origin specified as 2D
                    origin_arrays = origin
                else:
                    raise ValueError(
                        f"Origin array must have shape ({n_knots}, {ndim}) or ({1}, {ndim})"
                    )
            else:
                raise ValueError("Origin must be 1D or 2D array")

        # Process rotation matrices
        rotation_matrices = None
        if R is not None:
            R = np.asarray(R)
            if R.ndim == 2:
                if R.shape == (ndim, ndim):
                    # Constant rotation
                    rotation_matrices = R.reshape(1, ndim, ndim)
                else:
                    raise ValueError(
                        f"Rotation matrix must have shape ({ndim}, {ndim})"
                    )
            elif R.ndim == 3:
                if R.shape == (n_knots, ndim, ndim):
                    # Time-varying rotation
                    rotation_matrices = R
                elif R.shape == (1, ndim, ndim):
                    # Single rotation specified as 3D
                    rotation_matrices = R
                else:
                    raise ValueError(
                        f"Rotation array must have shape ({n_knots}, {ndim}, {ndim}) or (1, {ndim}, {ndim})"
                    )
            else:
                raise ValueError("Rotation must be 2D or 3D array")

            # Validate rotation matrices are orthogonal
            for i, rot_matrix in enumerate(rotation_matrices):
                if not self._is_orthogonal(rot_matrix):
                    raise ValueError(f"Rotation matrix at index {i} is not orthogonal")

        # Store arrays for the wrapper
        self._param_arrays = param_arrays
        self._origin_arrays = origin_arrays
        self._rotation_matrices = rotation_matrices

        # Initialize the base class
        super().__init__(
            units=units, origin=np.zeros(ndim), R=np.eye(ndim), **processed_params
        )

        # Create the time interpolation wrapper
        # self._setup_wrapper()

    def _setup_wrapper(self, **_):
        """Set up the time interpolation wrapper."""
        # Create a wrapped potential instance for the C layer
        temp_potential = self._potential_cls(
            units=self.units,
            origin=np.zeros(self.ndim),
            R=np.eye(self.ndim),
            **{
                k: v[0] if hasattr(v, "__len__") and len(v) > 1 else v
                for k, v in self._param_arrays.items()
            },
        )

        # Extract time knots in the potential's units
        if hasattr(self._time_knots, "unit"):
            time_knots_value = self._time_knots.decompose(self.units).value
        else:
            time_knots_value = self._time_knots

        # Create the Cython wrapper
        self.c_instance = TimeInterpolatedWrapper(
            temp_potential.c_instance,
            time_knots_value,
            self._param_arrays,
            self._origin_arrays,
            self._rotation_matrices,
            self._interp_kind,
        )

    @staticmethod
    def _is_orthogonal(matrix, rtol=1e-5, atol=1e-8):
        """Check if a matrix is orthogonal."""
        return np.allclose(
            matrix @ matrix.T, np.eye(matrix.shape[0]), rtol=rtol, atol=atol
        )

    def _energy(self, q, t):
        return self.c_instance.energy(q, t=t)

    def _gradient(self, q, t):
        return self.c_instance.gradient(q, t=t)

    def _density(self, q, t):
        return self.c_instance.density(q, t=t)

    def _hessian(self, q, t):
        return self.c_instance.hessian(q, t=t)

    @property
    def time_bounds(self):
        """Time bounds for interpolation (t_min, t_max)."""
        if hasattr(self.c_instance, "time_bounds"):
            bounds = self.c_instance.time_bounds
            if bounds is not None and hasattr(self._time_knots, "unit"):
                # Apply units if the original time knots had units
                return (
                    bounds[0] * self._time_knots.unit,
                    bounds[1] * self._time_knots.unit,
                )
            return bounds
        return None

    @property
    def wrapped_potential_class(self):
        """The wrapped potential class."""
        return self._potential_cls

    def replicate(self, **kwargs):
        """Create a copy of this potential with possibly different parameters."""
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
            interp_kind=self._interp_kind,
            units=self.units,
            origin=self._origin_arrays,
            R=self._rotation_matrices,
            **new_kwargs,
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: "
            f"{self._potential_cls.__name__} "
            f"(t_bounds={self.time_bounds}, "
            f"interp_kind='{self._interp_kind}')>"
        )
