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

            Naming convention for variables:
                im1 -> i-1
                im1_2 -> i-1/2
                ip1 -> i+1
                ip1_2 -> i+1/2

            `acceleration_func` should accept an array of position(s) and
            optionally a set of arguments specified by `func_args`.

            For details on the algorithm, see:
                http://en.wikipedia.org/wiki/Leapfrog_integration
                http://ursa.as.arizona.edu/~rad/phys305/ODE_III/node11.html

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

    def step(self, dt):
        """ Step forward the positions and velocities by the given timestep.

            Parameters
            ----------
            dt : numeric
                The timestep to move forward.
        """

        self.q_im1 += self.p_im1_2*dt
        a_i = self.acceleration(self.q_im1, *self._func_args)

        self.p_im1 += a_i*dt
        self.p_im1_2 += a_i*dt

        return self.q_im1, self.p_im1

    def _prime(self, dt):
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

        # here is where we scoot the velocity at t=0 to v(t+1/2)
        a_im1 = self.acceleration(self.q_im1, *self._func_args)
        self.p_im1_2 = self.p_im1 + a_im1*dt/2.

    def run(self, q_i, p_i, **time_spec):
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
            dt, t1 : (array_like, numeric)
                An array of timesteps dt and an initial time.
            t : array_like
                An array of times (dts = t[1:] - t[:-1])

        """

        self.q_im1 = np.atleast_2d(q_i)
        self.p_im1 = np.atleast_2d(p_i)

        # make sure they have the same shape
        if not self.q_im1.shape == self.p_im1.shape:
            raise ValueError("Shape of coordinates (q_i: {}) must match "
                             "momenta (p_i: {})".format(self.q_im1.shape,
                                                        self.p_im1.shape))

        times = _parse_time_specification(**time_spec)
        dts = times[1:]-times[:-1]
        ntimesteps = len(times)

        # prime the integrator so velocity is offset from coordinate by a
        #   half timestep
        self._prime(dts[0])

        # create the return arrays
        qs = np.zeros((ntimesteps,) + self.q_im1.shape, dtype=float)
        ps = np.zeros((ntimesteps,) + self.p_im1.shape, dtype=float)

        # Set first step to the initial conditions
        qs[0] = self.q_im1
        ps[0] = self.p_im1

        for ii,dt in enumerate(dts):
            q_i, p_i = self.step(dt)
            qs[ii+1] = q_i
            ps[ii+1] = p_i

        return times, qs, ps