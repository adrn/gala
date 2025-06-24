"""Wrapper around SciPy DOPRI853 integrator."""

from scipy.integrate import ode

from ..core import Integrator
from ..timespec import parse_time_specification

__all__ = ["DOPRI853Integrator"]


class DOPRI853Integrator(Integrator):
    r"""The Dormand-Prince 85(3) integration scheme.

    For Python integration, this class serves as a wrapper around Scipy's
    implementation of the integrator. See the Scipy documentation (`scipy.integrate.
    DOP853`) for more details and parameters. Note that the default tolerances for the
    Scipy integrator are quite poor for most applications, so you may want to set them
    to something like ``atol=1e-10`` and ``rtol=1e-10`` for better accuracy. Pass these
    to the class constructor as additional keyword arguments, e.g.::

        >>> integrator = DOPRI853Integrator(func, atol=1e-10, rtol=1e-10)

    For Cython integration, this class serves as a wrapper around a modified version of
    a C implementation of the integrator. The C implementation is based on the original
    Hairer and Wanner implementation, which is a C translation of the original Fortran
    code by Dormand and Prince. The available arguments for the Cython integrator are:

        - ``atol`` (float) - the minimum absolute error allowed in each integration
          variable (i.e. position and velocity components) per step
        - ``rtol`` (float) - the minimum relative error allowed in each integration
          variable (i.e. position and velocity components) per step
        - ``nmax`` (int) - the maximum number of integration steps
        - ``dt_max`` (float) - the maximum internal timestep used for integration
        - ``nstiff`` (int) - the number of steps to take before checking for stiffness
        - ``err_if_fail`` (bool) - raise an error if the integration fails
        - ``log_output`` (bool) - log any debug or additional error messages from the C
          integrator (useful for debugging failed integrations)

    New in version 1.10: The Cython implementation of the DOPRI853 integrator now uses
    the dense output functionality of the integrator, which allows for more efficient
    evaluation of the orbit on a dense grid of times. This is done by using internal
    interpolation to compute the orbit at the requested times, rather than using the
    original integration steps.

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
    **kwargs
        Additional keyword arguments to pass to the integrator.
        See the Scipy documentation for more details about Python integration options.
        For Cython integration, see the class docstring above for details.

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
        super().__init__(
            func, func_args, func_units, progress=progress, save_all=save_all
        )
        self._ode_kwargs = kwargs

    def __call__(self, w0, mmap=None, **time_spec):
        # generate the array of times
        times = parse_time_specification(self._func_units, **time_spec)
        n_steps = len(times) - 1

        w0, arr_w0, ws = self._prepare_ws(w0, mmap, n_steps)
        size_1d = 2 * self.ndim * self.norbits

        # need this to do resizing, and to handle func_args because there is some
        #   issue with the args stuff in scipy...
        def func_wrapper(t, x):
            x_ = x.reshape((2 * self.ndim, self.norbits))
            val = self.F(t, x_, *self._func_args)
            return val.reshape((size_1d,))

        self._ode = ode(func_wrapper, jac=None)
        self._ode = self._ode.set_integrator("dop853", **self._ode_kwargs)

        # create the return arrays
        if self.save_all:
            ws[:, 0] = arr_w0

        # make 1D
        arr_w0 = arr_w0.reshape((size_1d,))

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
