# coding: utf-8

""" 5th order Runge-Kutta integration. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np

# Project
from .core import Integrator
from .timespec import _parse_time_specification

__all__ = ["RK5Integrator"]

# These are the Dormand-Prince parameters for embedded Runge-Kutta methods
A = np.array([0.0, 0.2, 0.3, 0.6, 1.0, 0.875])
B = np.array([[        0.0,        0.0,         0.0,            0.0,        0.0],
            [      1./5.,        0.0,         0.0,            0.0,        0.0],
            [      3./40.,    9./40.,         0.0,            0.0,        0.0],
            [      3./10.,   -9./10.,       6./5.,            0.0,        0.0],
            [    -11./54.,     5./2.,    -70./27.,        35./27.,        0.0],
            [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.]])
C = np.array([37./378., 0., 250./621., 125./594., 0., 512./1771.])
D = np.array([2825./27648., 0., 18575./48384., 13525./55296., 277./14336., 1./4.])

class RK5Integrator(Integrator):
    r"""
    Initialize a 5th order Runge-Kutta integrator given a function for
    computing derivatives with respect to the independent variables. The
    function should, at minimum, take the independent variable as the
    first argument, and the coordinates as a single vector as the second
    argument. For notation and variable names, we assume this independent
    variable is time, t, and the coordinate vector is named x, though it
    could contain a mixture of coordinates and momenta for solving
    Hamilton's equations, for example.

    .. seealso::

        - http://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    **Example:** Harmonic oscillator

    Hamilton's equations are

    .. math::

        \dot{q} = \frac{\partial H}{\partial p}\\
        \dot{p} = -\frac{\partial H}{\partial q}

    The harmonic oscillator Hamiltonian is

    .. math::

        H(q,p) = \frac{1}{2}(p^2 + q^2)

    so that the equations of motion are given by

    .. math::

        \dot{q} = p\\
        \dot{p} = -q

    We then define a vector :math:`x = (q, p)`. The function passed in to
    the integrator should return the derivative of :math:`x` with respect to
    the independent variable,  :math:`\dot{x} = (\dot{q}, \dot{p})`, e.g.::

        def F(t,x):
            q,p = x.T
            return np.array([p,-q]).T

    To create an integrator object, just pass this function in to the
    constructor, and then we can integrate orbits from a given vector of
    initial conditions::

        integrator = RK5Integrator(F)
        times,ws = integrator.run(w0=[1.,0.], dt=0.1, nsteps=1000)

    .. note::

        Even though we only pass in a single vector of initial conditions,
        this gets promoted internally to a 2D array. This means the shape of
        the integrated orbit array will always be 3D. In this case, `ws` will
        have shape `(1001,1,2)`.

    Parameters
    ----------
    func : func
        A callable object that computes the phase-space coordinate
        derivatives with respect to the independent variable at a point
        in phase space.
    func_args : tuple (optional)
        Any extra arguments for the function.

    """

    def __init__(self, func, func_args=()):

        if not hasattr(func, '__call__'):
            raise ValueError("func must be a callable object, e.g., a function.")

        self.func = func
        self._func_args = func_args

    def step(self, t, w, dt):
        """ Step forward the vector w by the given timestep.

            Parameters
            ----------
            dt : numeric
                The timestep to move forward.
        """

        # Runge-Kutta Fehlberg formulas (see: Numerical Recipes)
        F = lambda t,w: self.func(t,w,*self._func_args)

        n = len(w)
        K = np.zeros((6,)+w.shape)
        K[0] = dt * F(t, w)
        K[1] = dt * F(t + A[1]*dt, w + B[1][0]*K[0])
        K[2] = dt * F(t + A[2]*dt, w + B[2][0]*K[0] + B[2][1]*K[1])
        K[3] = dt * F(t + A[3]*dt, w + B[3][0]*K[0] + B[3][1]*K[1] + B[3][2]*K[2])
        K[4] = dt * F(t + A[4]*dt, w + B[4][0]*K[0] + B[4][1]*K[1] + B[4][2]*K[2] + B[4][3]*K[3])
        K[5] = dt * F(t + A[5]*dt, w + B[5][0]*K[0] + B[5][1]*K[1] + B[5][2]*K[2] + B[5][3]*K[3] + B[5][4]*K[4])

        # shift
        dw = np.zeros_like(w)
        for i in range(6):
            dw = dw + C[i]*K[i]

        return w + dw

    def run(self, w0, **time_spec):
        """
        Run the integrator starting at the given coordinates and momenta
        (or velocities) and a time specification. The initial conditions
        `w0` should have shape `(nparticles, ndim)` or `(ndim,)` for a
        single orbit.

        There are a few combinations of keyword arguments accepted for
        specifying the timestepping. For example, you can specify a fixed
        timestep (`dt`) and a number of steps (`nsteps`), or an array of
        times. See **Other Parameters** below for more information.

        Parameters
        ==========
        w0 : array_like
            Initial conditions.

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

        w0 = np.atleast_2d(w0)
        nparticles, ndim = w0.shape

        # generate the array of times
        times = _parse_time_specification(**time_spec)
        nsteps = len(times)-1
        dt = times[1]-times[0]

        # create the return arrays
        ws = np.zeros((nsteps+1,) + w0.shape, dtype=float)

        # Set first step to the initial conditions
        ws[0] = w0
        w = w0.copy()
        for ii in range(1,nsteps+1):
            w = self.step(times[ii], w, dt)
            ws[ii] = w

        return times, ws

'''

class RK5AdaptiveIntegrator(Integrator):

    def __init__(self, func, func_args=()):
        """ Initialize a 5th order Runge-Kutta integrator given a function for
            computing ...TODO

            TODO
            `acceleration_func` should accept an array of position(s) and
            optionally a set of arguments specified by `func_args`.

            For details on the algorithm, see:


            Parameters
            ----------
            func : func
                A callable object that computes the TODO at a point
                in phase space.
            func_args : tuple (optional)
                Any extra arguments for the function.

        """

        if not hasattr(func, '__call__'):
            raise ValueError("func must be a callable object, e.g., a function.")

        self.func = func
        self._func_args = func_args

    def step(self, t, x, dt):
        """ Step forward the vector x by the given timestep.

            Parameters
            ----------
            dt : numeric
                The timestep to move forward.
        """

        # Runge-Kutta Fehlberg formulas (see: Numerical Recipes)
        F = lambda t,x: self.func(t,x,*self._func_args)

        n = len(x)
        K = np.zeros((6,)+x.shape)
        K[0] = dt * F(t, x)
        K[1] = dt * F(t + A[1]*dt, x + B[1][0]*K[0])
        K[2] = dt * F(t + A[2]*dt, x + B[2][0]*K[0] + B[2][1]*K[1])
        K[3] = dt * F(t + A[3]*dt, x + B[3][0]*K[0] + B[3][1]*K[1] + B[3][2]*K[2])
        K[4] = dt * F(t + A[4]*dt, x + B[4][0]*K[0] + B[4][1]*K[1] + B[4][2]*K[2] + B[4][3]*K[3])
        K[5] = dt * F(t + A[5]*dt, x + B[5][0]*K[0] + B[5][1]*K[1] + B[5][2]*K[2] + B[5][3]*K[3] + B[5][4]*K[4])

        # Initialize arrays {dy} and {E}
        E = np.zeros((n))
        dx = np.zeros((n))

        # Compute solution increment {dy}
        for i in range(6):
            dx = dx + C[i]*K[i]
            E = E + (C[i] - D[i])*K[i]

        _t = t + dt
        _x = x + dx

        # Compute RMS error e
        e = np.sqrt(sum(E**2)/n)

        return _t, _x, np.max(e)

    def run(self, q_i, p_i, nsteps, dt0, t0=0., tol=1E-6):
        """ TODO

            There are a few combinations of keyword arguments accepted for
            specifying the timestepping. For example, you can specify a fixed
            timestep (`dt`) and a number of steps (`nsteps`), or an array of
            times. See `kwargs` below for more information.

            Parameters
            ----------
            q_i : array_like
                Coordinate initial conditions.
            p_i : array_like
                Initial conditions for the momenta (or velocities) conjugate
                to the coordinates.

            kwargs
            ------
            dt, nsteps[, t1] : (numeric, int[, numeric])
                A fixed timestep dt and a number of steps to run for.
            dt, t1, t2 : (numeric, numeric, numeric)
                A fixed timestep dt, an initial time, and a final time.
            t : array_like
                An array of times (dt = t[1] - t[0])

        """

        # if _dt < 0.:
        #     self.p_im1 *= -1.
        #     dt = np.abs(_dt)
        # else:
        #     dt = _dt

        q_i = np.atleast_2d(q_i)
        p_i = np.atleast_2d(p_i)
        x = np.hstack((q_i, p_i))
        t = t0

        nparticles, ndim = q_i.shape

        # create the return arrays
        qs = np.zeros((nsteps+1,) + q_i.shape, dtype=float)
        ps = np.zeros((nsteps+1,) + p_i.shape, dtype=float)
        times = np.zeros((nsteps+1,))

        # Set first step to the initial conditions
        qs[0] = q_i
        ps[0] = p_i
        times[0] = t0

        dt = dt0
        e = 1E10
        for ii in range(1,nsteps+1):
            tt,xx,e = self.step(t,x,dt)
            _t = tt
            _x = xx
            while e >= tol:
                # Compute next step size (decrease or increase)
                if e == 0.0:
                    break
                else:
                    dt = 0.9 * dt * (tol/e)**0.2

                _t,_x,e = self.step(t,x,dt)
            else:
                t = _t
                x = _x

            times[ii] = t
            qs[ii] = x[...,:ndim]
            ps[ii] = x[...,ndim:]

        return times, qs, ps
        # if _dt < 0:
        #     return times, qs, -ps
        # else:
        #     return times, qs, ps
'''