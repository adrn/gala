"""Leapfrog integration."""

import numpy as np

from ..core import Integrator
from ..timespec import parse_time_specification

__all__ = ["LeapfrogIntegrator"]


class LeapfrogIntegrator(Integrator):
    r"""
    Symplectic leapfrog integrator for Hamiltonian systems.

    The leapfrog integrator is a second-order symplectic method that is
    particularly well-suited for integrating Hamiltonian systems over long
    time periods. It conserves energy exactly for linear systems and
    approximately for nonlinear systems, with bounded energy errors.

    The method alternately updates positions and momenta (velocities) in a
    "leapfrog" pattern, where velocities are evaluated at half-integer
    timesteps relative to positions. This staggered evaluation is what
    gives the method its symplectic properties.

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
    The leapfrog method uses the following update scheme:

    .. math::

        v_{i+1/2} &= v_{i-1/2} + a_i \Delta t \\
        x_{i+1} &= x_i + v_{i+1/2} \Delta t

    where :math:`a_i = F(t_i, x_i, v_{i-1/2})` is the acceleration at
    position :math:`x_i`.

    The integrator automatically handles the initial half-step offset for
    velocities by computing :math:`v_{1/2} = v_0 + a_0 \Delta t / 2` from
    the initial conditions.

    Advantages:
        * Symplectic (preserves phase-space structure)
        * Time-reversible
        * Excellent long-term energy conservation
        * Computationally efficient (one force evaluation per step)

    Disadvantages:
        * Only second-order accurate
        * Requires fixed timesteps
        * Less accurate than higher-order methods for smooth problems

    References
    ----------
    * Verlet, L. (1967). Computer "experiments" on classical fluids. Physical
      Review, 159(1), 98-103.
    * Leimkuhler, B. & Reich, S. (2004). Simulating Hamiltonian Dynamics.
      Cambridge University Press.

    Examples
    --------
    Simple harmonic oscillator with Hamiltonian :math:`H = \\frac{1}{2}(p^2 + q^2)`:

    .. code-block:: python

        def derivs(t, w):
            q, p = w[0], w[1]  # position, momentum
            return np.array([p, -q])  # [dq/dt, dp/dt]


        integrator = LeapfrogIntegrator(derivs)
        orbit = integrator(w0=[1.0, 0.0], dt=0.1, n_steps=1000)

    The derivative function must return an array where the first half
    contains position derivatives (velocities) and the second half contains
    momentum derivatives (accelerations).
    """

    def step(self, t, x_im1, v_im1_2, dt):
        """
        Advance the integration by one timestep using the leapfrog scheme.

        This method performs a single leapfrog step, updating positions
        and velocities according to the symplectic leapfrog algorithm.

        Parameters
        ----------
        t : float
            Current time.
        x_im1 : :class:`~numpy.ndarray`
            Position at the previous timestep, shape ``(ndim, norbits)``.
        v_im1_2 : :class:`~numpy.ndarray`
            Velocity at the previous half-timestep, shape ``(ndim, norbits)``.
        dt : float
            Integration timestep.

        Returns
        -------
        x_i : :class:`~numpy.ndarray`
            Updated position at the current timestep.
        v_i : :class:`~numpy.ndarray`
            Velocity at the current timestep (synchronized with position).
        v_ip1_2 : :class:`~numpy.ndarray`
            Velocity at the next half-timestep, ready for the next integration step.

        Notes
        -----
        The leapfrog step consists of:
        1. Update position: :math:`x_i = x_{i-1} + v_{i-1/2} \\Delta t`
        2. Compute force: :math:`F_i = F(t, x_i, v_{i-1/2})`
        3. Update velocity: :math:`v_{i+1/2} = v_{i-1/2} + a_i \\Delta t`
        4. Compute synchronized velocity: :math:`v_i = (v_{i-1/2} + v_{i+1/2})/2`
        """

        x_i = x_im1 + v_im1_2 * dt
        F_i = self.F(t, np.vstack((x_i, v_im1_2)), *self._func_args)
        a_i = F_i[self.ndim :]

        v_i = v_im1_2 + a_i * dt / 2
        v_ip1_2 = v_i + a_i * dt / 2

        return x_i, v_i, v_ip1_2

    def _init_v(self, t, w0, dt):
        """
        Leapfrog updates the velocities offset a half-step from the
        position updates. If we're given initial conditions aligned in
        time, e.g. the positions and velocities at the same 0th step,
        then we have to initially scoot the velocities forward by a half
        step to prime the integrator.

        Parameters
        ----------
        dt : numeric
            The first timestep.
        """

        # here is where we scoot the velocity at t=t1 to v(t+1/2)
        F0 = self.F(t.copy(), w0.copy(), *self._func_args)
        a0 = F0[self.ndim :]
        return w0[self.ndim :] + a0 * dt / 2.0

    def __call__(self, w0, mmap=None, **time_spec):
        # generate the array of times
        times = parse_time_specification(self._func_units, **time_spec)
        n_steps = len(times) - 1
        dt = times[1] - times[0]

        w0_obj, w0, ws = self._prepare_ws(w0, mmap, n_steps)
        x0 = w0[: self.ndim]

        # prime the integrator so velocity is offset from coordinate by a
        #   half timestep
        v_im1_2 = self._init_v(times[0], w0, dt)
        x_im1 = x0

        if self.save_all:
            ws[:, 0] = w0

        range_ = self._get_range_func()
        for ii in range_(1, n_steps + 1):
            x_i, v_i, v_ip1_2 = self.step(times[ii], x_im1, v_im1_2, dt)

            slc = (ii, slice(None)) if self.save_all else (slice(None),)
            ws[(slice(None, self.ndim), *slc)] = x_i
            ws[(slice(self.ndim, None), *slc)] = v_i
            x_im1, v_im1_2 = x_i, v_ip1_2

        if not self.save_all:
            times = times[-1:]

        return self._handle_output(w0_obj, times, ws)
