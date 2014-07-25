# coding: utf-8

""" Wrapper around SciPy vode integrator. """

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

__all__ = ["AdaptiveVODEIntegrator"]

class AdaptiveVODEIntegrator(Integrator):
    r"""
    This provides a wrapper around `Scipy`'s implementation of the
    Variable-coefficient Ordinary Differential Equation solver.

    .. seealso::

        - http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html

    To set the tolerance...TODO

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

    def run(self, w0, t2, t1=0.):
        """
        Run the integrator starting at the given coordinates and momenta
        (or velocities) and a time specification. The initial conditions
        `w0` should have shape `(nparticles, ndim)` or `(ndim,)` for a
        single orbit.

        This integrator uses an adaptive scheme and so you only have to
        specify the final time to integrate to.

        Parameters
        ==========
        w0 : array_like
            Initial conditions.
        t2 : numeric
            End time.
        t1 : numeric (optional)
            Start time. Assumes `t1=0` if none provided.

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

        # need this to do resizing, and to handle func_args because there is some
        #   issue with the args stuff in scipy...
        def func_wrapper(t,x):
            _x = x.reshape((nparticles,ndim))
            return self.func(t,_x,*self._func_args).reshape((nparticles*ndim,))

        self._ode = ode(func_wrapper, jac=None)
        self._ode = self._ode.set_integrator('vode', **self._ode_kwargs)

        # make 1D
        w0 = w0.reshape((nparticles*ndim,))

        # set the initial conditions
        self._ode.set_initial_value(w0, t1)

        # create the return arrays
        ws = [w0]
        ts = [t1]

        # Integrate the ODE(s) to the final time
        while self._ode.successful() and self._ode.t < t2:
            self._ode.integrate(t2, step=True)
            ts.append(self._ode.t)
            ws.append(self._ode.y)

        ts = np.array(ts)
        ws = np.array(ws)

        if not self._ode.successful():
            raise RuntimeError("ODE integration failed!")

        return ts, ws.reshape((nsteps+1,nparticles,ndim))