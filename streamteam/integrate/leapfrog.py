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

__all__ = ["LeapfrogIntegrator"]

class LeapfrogIntegrator(Integrator):

    def __init__(self, acceleration_func, func_args=()):
        """ Initialize a Leapfrog integrator given a function for computing
            accelerations.

            `acceleration_func` should accept an array of position(s) and
            optionally a set of arguments specified by `func_args`.

            For details on the algorithm, see:
                http://en.wikipedia.org/wiki/Leapfrog_integration
                http://ursa.as.arizona.edu/~rad/phys305/ODE_III/node11.html

            Naming convention for variables:
                im1 -> i-1
                im1_2 -> i-1/2
                ip1 -> i+1
                ip1_2 -> i+1/2

            Parameters
            ----------
            acceleration_func : func
                A callable object that computes the acceleration at a point
                in phase space.
            func_args : tuple (optional)
                Any extra arguments for the acceleration function.

        """

        if not hasattr(acceleration_func, '__call__'):
            raise ValueError("acceleration_func must be a callable object, "
                        "e.g. a function, that evaluates the acceleration "
                        "at a given position.")

        self.acceleration = acceleration_func
        self._func_args = func_args

    def step(self, t, x_im1, v_im1_2, dt):
        """ Step forward the positions and velocities by the given timestep.

            Parameters
            ----------
            dt : numeric
                The timestep to move forward.
        """

        x_i = x_im1 + v_im1_2*dt
        a_i = self.acceleration(t, x_i, *self._func_args)

        v_i = v_im1_2 + a_i*dt/2
        v_ip1_2 = v_i + a_i*dt/2

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
        a0 = self.acceleration(t, x, *self._func_args)
        v_1_2 = v + a0*dt/2.

        return v_1_2

    def run(self, w0, **time_spec):
        """ Run the integrator starting at the given coordinates and momenta
            (or velocities) and a time specification. The initial conditions
            `q`,`p` should each have shape `(nparticles, ndim)`.
            For example, for 100 particles in 3D cartesian coordinates, the
            coordinate (`q`) array should have shape (100,3) and the momentum
            (`p`) array should also have shape (100,3). For a single particle,
            1D arrays are promoted to 2D -- e.g., a coordinate array with shape
            (3,) is interpreted as a single particle in 3 dimensions and
            promoted to an array with shape (1,3).

            There are a few combinations of keyword arguments accepted for
            specifying the timestepping. For example, you can specify a fixed
            timestep (`dt`) and a number of steps (`nsteps`), or an array of
            times. See `kwargs` below for more information.

            Parameters
            ----------
            w0 : array_like
                Initial conditions

            kwargs
            ------
            dt, nsteps[, t1] : (numeric, int[, numeric])
                A fixed timestep dt and a number of steps to run for.
            dt, t1, t2 : (numeric, numeric, numeric)
                A fixed timestep dt, an initial time, and a final time.
            t : array_like
                An array of times (dt = t[1] - t[0])

        """

        w0 = np.atleast_2d(w0)
        nparticles, ndim = w0.shape

        if ndim%2 != 0:
            raise ValueError("Dimensionality must be even.")

        # dimensionality of positions,velocities
        self.ndim = ndim
        self.ndim_xv = self.ndim//2

        x0 = w0[...,:self.ndim_xv]
        v0 = w0[...,self.ndim_xv:]

        # generate the array of times
        times = _parse_time_specification(**time_spec)
        nsteps = len(times)-1
        _dt = times[1]-times[0]

        if _dt < 0.:
            v0 *= -1.
            dt = np.abs(_dt)
        else:
            dt = _dt

        # prime the integrator so velocity is offset from coordinate by a
        #   half timestep
        v_im1_2 = self._init_v(times[0], w0, dt)
        x_im1 = x0

        # create the return arrays
        ws = np.zeros((nsteps+1,) + w0.shape, dtype=float)
        ws[0,:,:self.ndim_xv] = x0
        ws[0,:,self.ndim_xv:] = v0
        for ii in range(1,nsteps+1):
            x_i, v_i, v_ip1_2 = self.step(times[ii], x_im1, v_im1_2, dt)
            ws[ii,:,:self.ndim_xv] = x_i
            ws[ii,:,self.ndim_xv:] = v_i
            x_im1, v_im1_2 = x_i, v_ip1_2

        if _dt < 0:
            ws[...,self.ndim_xv:] *= -1.

        return times, ws
