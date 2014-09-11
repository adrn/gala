# coding: utf-8

""" Leapfrog integration. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

# Project
from .core import Integrator
from .timespec import _parse_time_specification

__all__ = ["LeapfrogIntegrator"]


class LeapfrogIntegrator(Integrator):
    r"""
    Initialize a Leapfrog integrator given a function for computing
    accelerations. `acceleration_func` should accept an array of
    position(s) and optionally a set of arguments specified by
    `func_args`.

    .. seealso::

        - http://en.wikipedia.org/wiki/Leapfrog_integration
        - http://ursa.as.arizona.edu/~rad/phys305/ODE_III/node11.html

    Naming convention for variables::

        im1 = i-1
        im1_2 = i-1/2
        ip1 = i+1
        ip1_2 = i+1/2

    **Example:** Harmonic oscillator

    Using `x` as our coordinate variable, we want to numerically solve for
    an orbit in the potential

    .. math::

        \Phi = \frac{1}{2}x^2

    The acceleration is then just

    .. math::

        a = -\frac{\partial \Phi}{\partial x} = -x

    We define a function that computes this acceleration at any given
    time, `t`, and position, `x`::

        def acceleration(t,x):
            return -x

    .. note::

        The force here is not time dependent, but the acceleration function
        always has to accept the independent variable (e.g., time) as the
        first argument.

    To create an integrator object, just pass this acceleration function in
    to the constructor, and then we can integrate orbits from a given vector
    of initial conditions::

        integrator = LeapfrogIntegrator(acceleration)
        times,ws = integrator.run(w0=[1.,0.], dt=0.1, nsteps=1000)

    .. note::

        Even though we only pass in a single vector of initial conditions,
        this gets promoted internally to a 2D array. This means the shape of
        the integrated orbit array will always be 3D. In this case, `ws` will
        have shape `(1001,1,2)`.

    Parameters
    ----------
    acceleration_func : func
        A callable object that computes the acceleration at a point
        in phase space.
    func_args : tuple (optional)
        Any extra arguments for the acceleration function.

    """
    def __init__(self, acceleration_func, func_args=()):
        if not hasattr(acceleration_func, '__call__'):
            raise ValueError("acceleration_func must be a callable object, "
                             "e.g. a function, that evaluates the acceleration "
                             "at a given position.")

        self.acceleration = acceleration_func
        self._func_args = func_args

    def step(self, t, x_im1, v_im1_2, dt):
        """
        Step forward the positions and velocities by the given timestep.

        Parameters
        ----------
        dt : numeric
            The timestep to move forward.
        """

        x_i = x_im1 + v_im1_2 * dt
        a_i = self.acceleration(t, x_i, *self._func_args)

        v_i = v_im1_2 + a_i * dt / 2
        v_ip1_2 = v_i + a_i * dt / 2

        return x_i, v_i, v_ip1_2

    def _init_v(self, t, w0, dt):
        """ Leapfrog updates the velocities offset a half-step from the
            position updates. If we're given initial conditions aligned in
            time, e.g. the positions and velocities at the same 0th step,
            then we have to initially scoot the velocities forward by a half
            step to prime the integrator.

            Parameters
            ----------
            dt : numeric
                The first timestep.
        """

        x = w0[...,:self.ndim_xv]
        v = w0[...,self.ndim_xv:]

        # here is where we scoot the velocity at t=t1 to v(t+1/2)
        a0 = self.acceleration(t.copy(), x.copy(), *self._func_args)
        v_1_2 = v + a0*dt/2.

        return v_1_2

    def run(self, w0, mmap=None, **time_spec):
        """
        Run the integrator starting at the given coordinates and momenta
        (velocities) and a time specification. The initial conditions
        `w0` should have shape `(nparticles, 2*ndim)`. For example, for
        100 orbits in 3D cartesian coordinates, the initial condition
        array should have shape `(100,3)`. For a single orbit,
        1D arrays are promoted to 2D -- e.g., a coordinate array with shape
        (6,) is interpreted as a single orbit in 3 dimensions and
        promoted to an array with shape (1,6).

        There are a few combinations of keyword arguments accepted for
        specifying the timestepping. For example, you can specify a fixed
        timestep (`dt`) and a number of steps (`nsteps`), or an array of
        times. See **Other Parameters** below for more information.

        Parameters
        ==========
        w0 : array_like
            Initial conditions
        mmap : None, array_like (optional)
            Option to write integration output to a memory-mapped array so the memory
            usage doesn't explode. Must pass in a memory-mapped array, e.g., from
            `numpy.memmap`.

        Other Parameters
        ================
        dt, nsteps[, t1] : (numeric, int[, numeric])
            A fixed timestep dt and a number of steps to run for.
        dt, t1, t2 : (numeric, numeric, numeric)
            A fixed timestep dt, an initial time, and a final time.
        t : array_like
            An array of times (dt = t[1] - t[0])

        Returns
        =======
        times : array_like
            An array of times.
        w : array_like
            The array of positions and momenta (velocities) at each time in
            the time array. This array has shape `(Ntimes,Norbits,Ndim)`.

        """

        # generate the array of times
        times = _parse_time_specification(**time_spec)
        nsteps = len(times) - 1
        _dt = times[1] - times[0]

        w0, ws = self._prepare_ws(w0, mmap, nsteps)
        x0 = w0[:,:self.ndim_xv]
        v0 = w0[:,self.ndim_xv:]

        if (self.ndim % 2) != 0:
            raise ValueError("Dimensionality must be even.")

        if _dt < 0.:
            v0 *= -1.
            dt = np.abs(_dt)
        else:
            dt = _dt

        # prime the integrator so velocity is offset from coordinate by a
        #   half timestep
        v_im1_2 = self._init_v(times[0], w0, dt)
        x_im1 = x0

        ws[0] = w0
        for ii in range(1,nsteps+1):
            x_i, v_i, v_ip1_2 = self.step(times[ii], x_im1, v_im1_2, dt)
            ws[ii,:,:self.ndim_xv] = x_i
            ws[ii,:,self.ndim_xv:] = v_i
            x_im1, v_im1_2 = x_i, v_ip1_2

        if _dt < 0:
            ws[...,self.ndim_xv:] *= -1.

        return times, ws
