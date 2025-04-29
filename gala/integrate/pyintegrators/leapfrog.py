"""Leapfrog integration."""

# Third-party
import numpy as np

# Project
from ..core import Integrator
from ..timespec import parse_time_specification

__all__ = ["LeapfrogIntegrator"]


class LeapfrogIntegrator(Integrator):
    r"""
    A symplectic, Leapfrog integrator.

    Given a function for computing time derivatives of the phase-space
    coordinates, this object computes the orbit at specified times.

    .. seealso::

        - http://en.wikipedia.org/wiki/Leapfrog_integration
        - http://ursa.as.arizona.edu/~rad/phys305/ODE_III/node11.html

    Naming convention for variables::

        im1 = i-1
        im1_2 = i-1/2
        ip1 = i+1
        ip1_2 = i+1/2

    Examples
    --------

    Using ``q`` as our coordinate variable and ``p`` as the conjugate
    momentum, we want to numerically solve for an orbit in the
    potential (Hamiltonian)

    .. math::

        \Phi &= \frac{1}{2}q^2\\
        H(q, p) &= \frac{1}{2}(p^2 + q^2)


    In this system,

    .. math::

        \dot{q} &= \frac{\partial \Phi}{\partial p} = p \\
        \dot{p} &= -\frac{\partial \Phi}{\partial q} = -q


    We will use the variable ``w`` to represent the full phase-space vector,
    :math:`w = (q, p)`. We define a function that computes the time derivates
    at any given time, ``t``, and phase-space position, ``w``::

        def F(t, w):
            dw = [w[1], -w[0]]
            return dw

    .. note::

        The force here is not time dependent, but this function always has
        to accept the independent variable (e.g., time) as the
        first argument.

    To create an integrator object, just pass this acceleration function in
    to the constructor, and then we can integrate orbits from a given vector
    of initial conditions::

        integrator = LeapfrogIntegrator(acceleration)
        times, ws = integrator(w0=[1., 0.], dt=0.1, n_steps=1000)

    .. note::

        When integrating a single vector of initial conditions, the return
        array will have 2 axes. In the above example, the returned array will
        have shape ``(2, 1001)``. If an array of initial conditions are passed
        in, the return array will have 3 axes, where the last axis is for the
        individual orbits.

    Parameters
    ----------
    func : func
        A callable object that computes the phase-space time derivatives
        at a time and point in phase space.
    func_args : tuple (optional)
        Any extra arguments for the derivative function.
    func_units : `~gala.units.UnitSystem` (optional)
        If using units, this is the unit system assumed by the
        integrand function.

    """

    def step(self, t, x_im1, v_im1_2, dt):
        """
        Step forward the positions and velocities by the given timestep.

        Parameters
        ----------
        dt : numeric
            The timestep to move forward.
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
        v_1_2 = w0[self.ndim :] + a0 * dt / 2.0

        return v_1_2

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

            if self.save_all:
                slc = (ii, slice(None))
            else:
                slc = (slice(None),)
            ws[(slice(None, self.ndim),) + slc] = x_i
            ws[(slice(self.ndim, None),) + slc] = v_i
            x_im1, v_im1_2 = x_i, v_ip1_2

        if not self.save_all:
            times = times[-1:]

        return self._handle_output(w0_obj, times, ws)
