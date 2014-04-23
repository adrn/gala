# coding: utf-8

""" Wrapper around SciPy DOPRI853 integrator. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from scipy.integrate import ode

# Project
from .core import Integrator
from .timespec import _parse_time_specification

__all__ = ["DOPRI853Integrator"]

class DOPRI853Integrator(Integrator):

    def __init__(self, func, func_args=(), **kwargs):
        """ ...TODO

            Parameters
            ----------
            func : func
                A callable object that computes the phase-space coordinate
                derivatives with respect to the independent variable at a point
                in phase space.
            func_args : tuple (optional)
                Any extra arguments for the function.

        """

        if not hasattr(func, '__call__'):
            raise ValueError("func must be a callable object, e.g., a function.")

        self.func = func
        self._ode = ode(self.func, jac=None).set_integrator('dop853', **kwargs)
        self._ode.set_f_params(*func_args)

    def run(self, x_i, **time_spec):
        """ Run the integrator starting at the given coordinates and momenta
            (or velocities) and a time specification. The initial conditions
            `x` should each have shape `(nparticles, ndim)`.

            There are a few combinations of keyword arguments accepted for
            specifying the timestepping. For example, you can specify a fixed
            timestep (`dt`) and a number of steps (`nsteps`), or an array of
            times. See `kwargs` below for more information.

            Parameters
            ----------
            x0 : array_like
                Initial conditions.

            kwargs
            ------
            dt, nsteps[, t1] : (numeric, int[, numeric])
                A fixed timestep dt and a number of steps to run for.
            dt, t1, t2 : (numeric, numeric, numeric)
                A fixed timestep dt, an initial time, and a final time.

        """

        x_i = np.atleast_2d(x_i)
        nparticles, ndim = x_i.shape

        # generate the array of times
        times = _parse_time_specification(**time_spec)
        nsteps = len(times)-1
        dt = times[1]-times[0]

        # set the initial conditions
        self._ode.set_initial_value(x_i.T, times[0])

        # create the return arrays
        xs = np.zeros((nsteps+1,) + x_i.shape, dtype=float)
        xs[0] = x_i

        #for ii in range(1,nsteps+1):
        # Integrate the ODE(s) across each delta_t timestep
        k = 1
        while self._ode.successful() and k < (nsteps+1):
            self._ode.integrate(self._ode.t + dt)
            xs[k] = self._ode.y.T
            k += 1

        return times, xs