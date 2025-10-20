"""Leapfrog integration."""

from ..core import Integrator
from ..timespec import parse_time_specification

__all__ = ["Ruth4Integrator"]


class Ruth4Integrator(Integrator):
    r"""
    Fourth-order symplectic integrator using Ruth's method.

    This integrator implements a fourth-order symplectic integration scheme
    developed by Ruth (1983). It provides higher accuracy than the standard
    leapfrog method while preserving the symplectic structure of Hamiltonian
    systems, making it excellent for long-term orbital integrations.

    The method uses a composition of multiple leapfrog-like steps with
    carefully chosen coefficients to achieve fourth-order accuracy while
    maintaining symplecticity and time-reversibility.

    Parameters
    ----------
    func : callable
        A function that computes the phase-space coordinate derivatives.
        Must have signature ``func(t, w, *func_args)`` where ``t`` is time,
        ``w`` is the phase-space position array with shape ``(2*ndim, ...)``,
        and ``*func_args`` are additional arguments.
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
    The Ruth4 method uses the following composition coefficients:

    .. math::

        c_1 = c_4 &= \\frac{1}{2(2-2^{1/3})} \\\\
        c_2 = c_3 &= \\frac{1-2^{1/3}}{2(2-2^{1/3})} \\\\
        d_1 &= 0 \\\\
        d_2 &= \\frac{1}{2-2^{1/3}} \\\\
        d_3 &= \\frac{-2^{1/3}}{2-2^{1/3}} \\\\
        d_4 &= \\frac{1}{2-2^{1/3}}

    Each timestep consists of four substeps that collectively achieve
    fourth-order accuracy.

    Advantages:
        * Fourth-order accuracy (vs second-order for leapfrog)
        * Symplectic (preserves phase-space structure)
        * Time-reversible
        * Excellent long-term stability for Hamiltonian systems

    Disadvantages:
        * More expensive than leapfrog (4 force evaluations per step)
        * Requires fixed timesteps
        * Can be less stable than leapfrog for some stiff problems

    References
    ----------
    * Ruth, R. D. (1983). A canonical integration technique. IEEE Transactions
      on Nuclear Science, 30(4), 2669-2671.
    * Forest, E. & Ruth, R. D. (1990). Fourth-order symplectic integration.
      Physica D, 43(1), 105-117.

    Examples
    --------
    Simple harmonic oscillator with Hamiltonian :math:`H = \\frac{1}{2}(p^2 + q^2)`:

    .. code-block:: python

        def derivs(t, w):
            q, p = w[0], w[1]  # position, momentum
            return np.array([p, -q])  # [dq/dt, dp/dt]


        integrator = Ruth4Integrator(derivs)
        orbit = integrator(w0=[1.0, 0.0], dt=0.1, n_steps=1000)

    The derivative function must return an array where the first half
    contains position derivatives (velocities) and the second half contains
    momentum derivatives (accelerations).
    """

    # From: https://en.wikipedia.org/wiki/Symplectic_integrator
    _cs = [
        1 / (2 * (2 - 2 ** (1 / 3))),
        (1 - 2 ** (1 / 3)) / (2 * (2 - 2 ** (1 / 3))),
        (1 - 2 ** (1 / 3)) / (2 * (2 - 2 ** (1 / 3))),
        1 / (2 * (2 - 2 ** (1 / 3))),
    ]
    _ds = [
        0,
        1 / (2 - 2 ** (1 / 3)),
        -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
        1 / (2 - 2 ** (1 / 3)),
    ]

    def step(self, t, w, dt):
        """
        Step forward the positions and velocities by the given timestep.

        Parameters
        ----------
        dt : numeric
            The timestep to move forward.
        """

        w_i = w.copy()
        for cj, dj in zip(self._cs, self._ds):
            F_i = self.F(t, w_i, *self._func_args)
            a_i = F_i[self.ndim :]

            w_i[self.ndim :] += dj * a_i * dt
            w_i[: self.ndim] += cj * w_i[self.ndim :] * dt

        return w_i

    def __call__(self, w0, mmap=None, **time_spec):
        # generate the array of times
        times = parse_time_specification(self._func_units, **time_spec)
        n_steps = len(times) - 1
        dt = times[1] - times[0]

        w0_obj, w0, ws = self._prepare_ws(w0, mmap, n_steps=n_steps)

        # Set first step to the initial conditions
        if self.save_all:
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
