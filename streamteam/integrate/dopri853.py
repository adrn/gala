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
    """
    ...TODO

    Parameters
    ----------
    func : func
        A callable object that computes the phase-space coordinate
        derivatives with respect to the independent variable at a point
        in phase space.
    func_args : tuple (optional)
        Any extra arguments for the function.

    """

    def __init__(self, func, func_args=(), **kwargs):

        if not hasattr(func, '__call__'):
            raise ValueError("func must be a callable object, e.g., a function.")

        self.func = func
        self._func_args = func_args
        self._ode_kwargs = kwargs

    def run(self, w0, **time_spec):
        """ Run the integrator starting at the given coordinates and momenta
            (or velocities) and a time specification. The initial conditions
            `w` should each have shape `(nparticles, ndim)`.

            There are a few combinations of keyword arguments accepted for
            specifying the timestepping. For example, you can specify a fixed
            timestep (`dt`) and a number of steps (`nsteps`), or an array of
            times. See `kwargs` below for more information.

            Parameters
            ----------
            w0 : array_like
                Initial conditions.

            kwargs
            ------
            dt, nsteps[, t1] : (numeric, int[, numeric])
                A fixed timestep dt and a number of steps to run for.
            dt, t1, t2 : (numeric, numeric, numeric)
                A fixed timestep dt, an initial time, and a final time.

        """

        w0 = np.atleast_2d(w0)
        nparticles, ndim = w0.shape

        # need this to do resizing, and to handle func_args because there is some
        #   issue with the args stuff in scipy...
        def func_wrapper(t,x):
            _x = x.reshape((nparticles,ndim))
            return self.func(t,_x,*self._func_args).reshape((nparticles*ndim,))

        self._ode = ode(func_wrapper, jac=None)
        self._ode = self._ode.set_integrator('dop853', **self._ode_kwargs)

        # make 1D
        w0 = w0.reshape((nparticles*ndim,))

        # generate the array of times
        times = _parse_time_specification(**time_spec)
        nsteps = len(times)-1
        dt = times[1]-times[0]

        # set the initial conditions
        self._ode.set_initial_value(w0, times[0])

        # create the return arrays
        ws = np.zeros((nsteps+1,w0.size), dtype=float)
        ws[0] = w0

        # Integrate the ODE(s) across each delta_t timestep
        k = 1
        while self._ode.successful() and k < (nsteps+1):
            self._ode.integrate(self._ode.t + dt)
            ws[k] = self._ode.y
            k += 1

        if not self._ode.successful():
            raise RuntimeError("ODE integration failed!")

        return times, ws.reshape((nsteps+1,nparticles,ndim))