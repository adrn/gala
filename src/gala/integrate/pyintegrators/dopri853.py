"""Wrapper around SciPy DOPRI853 integrator."""

from scipy.integrate import ode

from ..core import Integrator
from ..timespec import parse_time_specification

__all__ = ["DOPRI853Integrator"]


class DOPRI853Integrator(Integrator):
    r"""
    Dormand-Prince 85(3) adaptive step-size integrator.

    This integrator implements the Dormand-Prince method, which is an explicit
    Runge-Kutta method with adaptive step-size control. It uses a 5th-order
    accurate formula for advancing the solution and an embedded 3rd-order
    formula for error estimation.

    For Python integration, this class wraps SciPy's implementation of the
    integrator. The default tolerances (``atol=1.49e-8``, ``rtol=1.49e-8``)
    may be too loose for many astronomical applications. Consider using
    tighter tolerances like ``atol=1e-10`` and ``rtol=1e-10`` for better
    accuracy::

        >>> integrator = DOPRI853Integrator(func, atol=1e-10, rtol=1e-10)

    For Cython integration, this class wraps a C implementation based on the
    original Hairer and Wanner code. The C version includes dense output
    functionality (new in v1.10) that allows efficient evaluation at arbitrary
    times through internal interpolation.

    Parameters
    ----------
    func : callable
        A function that computes the phase-space coordinate derivatives.
        Must have signature ``func(t, w, *func_args)`` where ``t`` is time,
        ``w`` is the phase-space position array, and ``*func_args`` are
        additional arguments.
    func_args : tuple, optional
        Additional arguments to pass to the derivative function.
    func_units : :class:`~gala.units.UnitSystem`, optional
        Unit system assumed by the integrand function.
    progress : bool, optional
        Display a progress bar during integration. Default is False.
    save_all : bool, optional
        Save the orbit at all timesteps. If False, only save the final state.
        Default is True.
    **kwargs
        Additional keyword arguments for the integrator:

        For Python (SciPy) integration:
            * ``atol`` (float) : Absolute tolerance (default: 1.49e-8)
            * ``rtol`` (float) : Relative tolerance (default: 1.49e-8)
            * ``nsteps`` (int) : Maximum number of steps (default: 500)
            * ``max_step`` (float) : Maximum step size (default: 0.0, no limit)
            * ``first_step`` (float) : Initial step size (default: 0.0, automatic)

        For Cython integration:
            * ``atol`` (float) : Absolute error tolerance per step
            * ``rtol`` (float) : Relative error tolerance per step
            * ``nmax`` (int) : Maximum number of integration steps
            * ``dt_max`` (float) : Maximum internal timestep
            * ``nstiff`` (int) : Steps before checking for stiffness
            * ``err_if_fail`` (bool) : Raise error if integration fails
            * ``log_output`` (bool) : Log debug messages from C integrator

    Notes
    -----
    The DOPRI853 method is well-suited for smooth problems where high accuracy
    is required. It automatically adjusts the step size to maintain the
    specified error tolerances, making it efficient for problems with varying
    time scales.

    References
    ----------
    * Dormand, J. R. & Prince, P. J. (1980). A family of embedded Runge-Kutta
      formulae. Journal of Computational and Applied Mathematics, 6(1), 19-26.
    * Hairer, E., Nørsett, S. P. & Wanner, G. (1993). Solving Ordinary
      Differential Equations I. Springer-Verlag.

    Examples
    --------
    Create an integrator with tight tolerances::

        >>> def derivs(t, w):
        ...     # Simple harmonic oscillator
        ...     return np.array([w[1], -w[0]])
        >>> integrator = DOPRI853Integrator(derivs, atol=1e-12, rtol=1e-12)
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
