"""5th order Runge-Kutta integration."""

import numpy as np

from ..core import Integrator
from ..timespec import parse_time_specification

__all__ = ["RK5Integrator"]

# These are the Dormand-Prince parameters for embedded Runge-Kutta methods
A = np.array([0.0, 0.2, 0.3, 0.6, 1.0, 0.875])
B = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
        [3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0, 0.0, 0.0],
        [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0, 0.0],
        [
            1631.0 / 55296.0,
            175.0 / 512.0,
            575.0 / 13824.0,
            44275.0 / 110592.0,
            253.0 / 4096.0,
        ],
    ]
)
C = np.array([37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0])
D = np.array(
    [
        2825.0 / 27648.0,
        0.0,
        18575.0 / 48384.0,
        13525.0 / 55296.0,
        277.0 / 14336.0,
        1.0 / 4.0,
    ]
)


class RK5Integrator(Integrator):
    r"""
    Fifth-order Runge-Kutta integrator with fixed timesteps.

    This integrator implements the classical fifth-order Runge-Kutta method
    (RK5) using the Dormand-Prince coefficients. It provides fifth-order
    accuracy for smooth problems with a fixed timestep, making it suitable
    for problems where high accuracy is needed and the solution varies
    smoothly in time.

    Unlike adaptive methods, this integrator uses a fixed timestep throughout
    the integration, which can be more predictable but may be less efficient
    for problems with varying time scales.

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

    Notes
    -----
    The RK5 method uses six function evaluations per timestep to achieve
    fifth-order accuracy. The update formula is:

    .. math::

        w_{n+1} = w_n + \\sum_{i=1}^{6} c_i k_i

    where the :math:`k_i` are intermediate slope estimates computed using
    the Dormand-Prince coefficients.

    Advantages:
        * Fifth-order accuracy for smooth problems
        * Stable and robust for most ODE systems
        * Predictable computational cost (6 function evaluations per step)

    Disadvantages:
        * Not symplectic (may not conserve energy for Hamiltonian systems)
        * Fixed timestep can be inefficient
        * More expensive per step than lower-order methods

    References
    ----------
    * Dormand, J. R. & Prince, P. J. (1980). A family of embedded Runge-Kutta
      formulae. Journal of Computational and Applied Mathematics, 6(1), 19-26.
    * Hairer, E., Nørsett, S. P. & Wanner, G. (1993). Solving Ordinary
      Differential Equations I. Springer-Verlag.

    Examples
    --------
    Integrate a simple harmonic oscillator:

    .. code-block:: python

        def derivs(t, w):
            return np.array([w[1], -w[0]])  # [dx/dt, dv/dt]


        integrator = RK5Integrator(derivs)
        orbit = integrator(w0=[1.0, 0.0], dt=0.01, n_steps=1000)
    """

    def step(self, t, w, dt):
        """
        Advance the integration by one timestep using the RK5 method.

        This method performs a single Runge-Kutta step using the classical
        fifth-order formula with Dormand-Prince coefficients.

        Parameters
        ----------
        t : float
            Current time.
        w : :class:`~numpy.ndarray`
            Current state vector with shape ``(2*ndim, norbits)``.
        dt : float
            Integration timestep.

        Returns
        -------
        w_new : :class:`~numpy.ndarray`
            Updated state vector at time ``t + dt``.

        Notes
        -----
        The method computes six intermediate slopes :math:`k_1, ..., k_6`
        and combines them with the Dormand-Prince weights to achieve
        fifth-order accuracy.
        """

        # Runge-Kutta Fehlberg formulas (see: Numerical Recipes)
        F = lambda t, w: self.F(t, w, *self._func_args)

        K = np.zeros((6, *w.shape))
        K[0] = dt * F(t, w)
        K[1] = dt * F(t + A[1] * dt, w + B[1][0] * K[0])
        K[2] = dt * F(t + A[2] * dt, w + B[2][0] * K[0] + B[2][1] * K[1])
        K[3] = dt * F(
            t + A[3] * dt, w + B[3][0] * K[0] + B[3][1] * K[1] + B[3][2] * K[2]
        )
        K[4] = dt * F(
            t + A[4] * dt,
            w + B[4][0] * K[0] + B[4][1] * K[1] + B[4][2] * K[2] + B[4][3] * K[3],
        )
        K[5] = dt * F(
            t + A[5] * dt,
            w
            + B[5][0] * K[0]
            + B[5][1] * K[1]
            + B[5][2] * K[2]
            + B[5][3] * K[3]
            + B[5][4] * K[4],
        )

        # shift
        dw = np.zeros_like(w)
        for i in range(6):
            dw += C[i] * K[i]

        return w + dw

    def __call__(self, w0, mmap=None, **time_spec):
        # generate the array of times
        times = parse_time_specification(self._func_units, **time_spec)
        n_steps = len(times) - 1
        dt = times[1] - times[0]

        w0_obj, w0, ws = self._prepare_ws(w0, mmap, n_steps=n_steps)

        if self.save_all:
            # Set first step to the initial conditions
            ws[:, 0] = w0
        w = w0.copy()
        range_ = self._get_range_func()
        for ii in range_(1, n_steps + 1):
            w = self.step(times[ii], w, dt)

            if self.save_all:
                ws[:, ii] = w

        if not self.save_all:
            ws = w
            times = times[-1:]

        return self._handle_output(w0_obj, times, ws)
