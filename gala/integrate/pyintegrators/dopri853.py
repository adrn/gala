"""Wrapper around SciPy DOPRI853 integrator."""

# Third-party
from scipy.integrate import ode

# Project
from ..core import Integrator
from ..timespec import parse_time_specification

__all__ = ["DOPRI853Integrator"]


class DOPRI853Integrator(Integrator):
    r"""
    This provides a wrapper around ``Scipy``'s implementation of the
    Dormand-Prince 85(3) integration scheme.

    .. seealso::

        - Numerical recipes (Dopr853)
        - http://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

    Parameters
    ----------
    func : callable
        A callable object that computes the phase-space coordinate
        derivatives with respect to the independent variable at a point
        in phase space.
    func_args : tuple (optional)
        Any extra arguments for the function.
    func_units : `~gala.units.UnitSystem` (optional)
        If using units, this is the unit system assumed by the
        integrand function.
    progress : bool (optional)
        Display a progress bar during integration.

    """

    def __init__(
        self,
        func,
        func_args=(),
        func_units=None,
        progress=False,
        save_all=True,
        **kwargs,
    ):
        super(DOPRI853Integrator, self).__init__(
            func, func_args, func_units, progress=progress, save_all=save_all
        )
        self._ode_kwargs = kwargs

    def __call__(self, w0, mmap=None, **time_spec):
        # generate the array of times
        times = parse_time_specification(self._func_units, **time_spec)
        n_steps = len(times) - 1

        w0, arr_w0, ws = self._prepare_ws(w0, mmap, n_steps)
        _size_1d = 2 * self.ndim * self.norbits

        # need this to do resizing, and to handle func_args because there is some
        #   issue with the args stuff in scipy...
        def func_wrapper(t, x):
            _x = x.reshape((2 * self.ndim, self.norbits))
            val = self.F(t, _x, *self._func_args)
            return val.reshape((_size_1d,))

        self._ode = ode(func_wrapper, jac=None)
        self._ode = self._ode.set_integrator("dop853", **self._ode_kwargs)

        # create the return arrays
        if self.save_all:
            ws[:, 0] = arr_w0

        # make 1D
        arr_w0 = arr_w0.reshape((_size_1d,))

        # set the initial conditions
        self._ode.set_initial_value(arr_w0, times[0])

        # Integrate the ODE(s) across each delta_t timestep
        range_ = self._get_range_func()
        for k in range_(1, n_steps + 1):
            self._ode.integrate(times[k])
            outy = self._ode.y

            if self.save_all:
                ws[:, k] = outy.reshape(2 * self.ndim, self.norbits)

            if not self._ode.successful():
                raise RuntimeError("ODE integration failed!")

        if not self.save_all:
            ws = outy.reshape(2 * self.ndim, 1, self.norbits)
            times = times[-1:]

        return self._handle_output(w0, times, ws)
