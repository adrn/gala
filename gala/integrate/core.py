"""Base class for integrators."""

from abc import ABCMeta, abstractmethod

import numpy as np
from astropy.utils.decorators import deprecated

from gala.units import DimensionlessUnitSystem, UnitSystem

__all__ = ["Integrator"]


class Integrator(metaclass=ABCMeta):
    """
    Abstract base class for numerical integrators.

    This class provides a common interface for different numerical integration
    schemes used to integrate orbits in gravitational potentials. All concrete
    integrator classes should inherit from this base class.

    Parameters
    ----------
    func : callable
        A function that computes the time derivatives of the phase-space
        coordinates. Should have the signature ``func(t, w, *func_args)``
        where ``t`` is the time, ``w`` is the current phase-space position,
        and ``*func_args`` are additional arguments.
    func_args : tuple, optional
        Additional arguments to pass to the derivative function. Default is ().
    func_units : :class:`~gala.units.UnitSystem`, optional
        The unit system assumed by the integrand function. If not provided,
        uses a dimensionless unit system.
    progress : bool, optional
        Whether to display a progress bar during integration. Requires the
        ``tqdm`` package. Default is False.
    save_all : bool, optional
        Whether to save the orbit at all integration timesteps. If False,
        only saves the final state. Default is True.

    Raises
    ------
    ValueError
        If ``func`` is not callable.
    ImportError
        If ``progress=True`` but the ``tqdm`` package is not installed.
    """

    def __init__(
        self,
        func,
        func_args=(),
        func_units=None,
        progress=False,
        save_all=True,
    ):
        if not callable(func):
            raise ValueError("func must be a callable object, e.g., a function.")

        self.F = func
        self._func_args = func_args

        if func_units is not None and not isinstance(
            func_units, DimensionlessUnitSystem
        ):
            func_units = UnitSystem(func_units)
        else:
            func_units = DimensionlessUnitSystem()
        self._func_units = func_units

        self.progress = bool(progress)
        self.save_all = save_all

    def _get_range_func(self):
        if self.progress:
            try:
                from tqdm import trange

                return trange
            except ImportError as e:
                msg = (
                    "tqdm must be installed to use progress=True when running "
                    f"{self.__class__.__name__}"
                )
                raise ImportError(msg) from e

        return range

    def _prepare_ws(self, w0, mmap, n_steps):
        """
        Decide how to make the return array. If ``mmap`` is False, this returns a full
        array of zeros, but with the correct shape as the output. If ``mmap`` is True,
        return a pointer to a memory-mapped array. The latter is particularly useful for
        integrating a large number of orbits or integrating a large number of time
        steps.
        """
        from ..dynamics import PhaseSpacePosition

        if not isinstance(w0, PhaseSpacePosition):
            w0 = PhaseSpacePosition.from_w(w0)

        arr_w0 = w0.w(self._func_units)

        self.ndim, self.norbits = arr_w0.shape
        self.ndim //= 2

        if self.save_all:
            return_shape = (2 * self.ndim, n_steps + 1, self.norbits)
        else:
            return_shape = (2 * self.ndim, self.norbits)

        if mmap is None:
            # create the return arrays
            ws = np.zeros(return_shape, dtype=float)

        else:
            if mmap.shape != return_shape:
                raise ValueError(
                    "Shape of memory-mapped array doesn't match expected shape of "
                    f"return array ({mmap.shape} vs {return_shape})"
                )

            if not mmap.flags.writeable:
                raise TypeError(
                    f"Memory-mapped array must be a writable mode, not '{mmap.mode}'"
                )

            ws = mmap

        return w0, arr_w0, ws

    def _handle_output(self, w0, t, w):
        """ """
        if w.shape[-1] == 1:
            w = w[..., 0]

        pos_unit = self._func_units["length"]
        t_unit = self._func_units["time"]
        vel_unit = pos_unit / t_unit

        from ..dynamics import Orbit

        return Orbit(
            pos=w[: self.ndim] * pos_unit,
            vel=w[self.ndim :] * vel_unit,
            t=t * t_unit,
        )

    @deprecated("1.9", alternative="Integrator call method")
    def run(self, w0, mmap=None, **time_spec):
        """Run the integrator starting from the specified phase-space position.

        .. deprecated:: 1.9
            Use the ``__call__`` method instead.
        """
        return self(w0, mmap=mmap, **time_spec)

    @abstractmethod
    def __call__(self, w0, mmap=None, **time_spec):
        """
        Run the integrator starting from the specified initial conditions.

        This method integrates the orbit forward in time from the given
        initial phase-space position according to the time specification.

        Parameters
        ----------
        w0 : :class:`~gala.dynamics.PhaseSpacePosition`
            Initial conditions for the integration.
        mmap : :class:`~numpy.ndarray`, optional
            A pre-allocated memory-mapped array to store the results.
            Must have the correct shape for the expected output.
        **time_spec
            Keyword arguments specifying the integration time. Accepted
            combinations include:

            * ``dt, n_steps[, t1]`` : Fixed timestep and number of steps
            * ``dt, t1, t2`` : Fixed timestep with start and end times
            * ``t`` : Array of specific times to integrate to

        Returns
        -------
        orbit : :class:`~gala.dynamics.Orbit`
            The integrated orbit containing positions, velocities, and times.

        Notes
        -----
        The time specification is parsed by
        :func:`~gala.integrate.timespec.parse_time_specification`. See that
        function's documentation for more details on the accepted formats.
        """
