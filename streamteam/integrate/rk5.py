# coding: utf-8

""" Leapfrog integration. """

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