""" Leapfrog integration. """

# Project
from ..core import Integrator
from ..timespec import parse_time_specification

__all__ = ["Ruth4Integrator"]


class Ruth4Integrator(Integrator):
    r"""
    A 4th order symplectic integrator.

    Given a function for computing time derivatives of the phase-space
    coordinates, this object computes the orbit at specified times.

    .. seealso::

        - https://en.wikipedia.org/wiki/Symplectic_integrator#A_fourth-order_example

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

        integrator = Ruth4Integrator(acceleration)
        times, ws = integrator.run(w0=[1., 0.], dt=0.1, n_steps=1000)

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

    def run(self, w0, mmap=None, **time_spec):

        # generate the array of times
        times = parse_time_specification(self._func_units, **time_spec)
        n_steps = len(times) - 1
        dt = times[1] - times[0]

        w0_obj, w0, ws = self._prepare_ws(w0, mmap, n_steps=n_steps)

        # Set first step to the initial conditions
        ws[:, 0] = w0
        w = w0.copy()
        range_ = self._get_range_func()
        for ii in range_(1, n_steps + 1):
            w = self.step(times[ii], w, dt)
            ws[:, ii] = w

        return self._handle_output(w0_obj, times, ws)
