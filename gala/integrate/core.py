""" Base class for integrators. """

from abc import ABCMeta, abstractmethod

# Third-party
import numpy as np
from astropy.utils.decorators import deprecated

# This project
from gala.units import UnitSystem, DimensionlessUnitSystem

__all__ = ["Integrator"]


class Integrator(metaclass=ABCMeta):
    def __init__(
        self,
        func,
        func_args=(),
        func_units=None,
        progress=False,
        store_all=True,
    ):
        if not hasattr(func, "__call__"):
            raise ValueError(
                "func must be a callable object, e.g., a function."
            )

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
        self.store_all = store_all

    def _get_range_func(self):
        if self.progress:
            try:
                from tqdm import trange
                return trange
            except ImportError:
                raise ImportError(
                    "tqdm must be installed to use progress=True when running "
                    f"{self.__class__.__name__}"
                )

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
        self.ndim = self.ndim // 2

        if self.store_all:
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

        orbit = Orbit(
            pos=w[:self.ndim] * pos_unit,
            vel=w[self.ndim:] * vel_unit,
            t=t * t_unit,
        )
        return orbit

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
        Run the integrator starting from the specified phase-space position.
        The initial conditions ``w0`` should be a
        `~gala.dynamics.PhaseSpacePosition` instance.

        There are a few combinations of keyword arguments accepted for
        specifying the timestepping. For example, you can specify a fixed
        timestep (``dt``) and a number of steps (``n_steps``), or an array of
        times::

            dt, n_steps[, t1] : (numeric, int[, numeric])
                A fixed timestep dt and a number of steps to run for.
            dt, t1, t2 : (numeric, numeric, numeric)
                A fixed timestep dt, an initial time, and a final time.
            t : array-like
                An array of times to solve on.

        Parameters
        ----------
        w0 : `~gala.dynamics.PhaseSpacePosition`
            Initial conditions.
        **time_spec
            Timestep information passed to
            `~gala.integrate.time_spec.parse_time_specification`.

        Returns
        -------
        orbit : `~gala.dynamics.Orbit`

        """
        pass
