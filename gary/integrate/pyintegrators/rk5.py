# coding: utf-8

""" 5th order Runge-Kutta integration. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

# Project
from ..core import Integrator
from ..timespec import _parse_time_specification

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

    Parameters
    ----------
    func : func
        A callable object that computes the phase-space coordinate
        derivatives with respect to the independent variable at a point
        in phase space.
    func_args : tuple (optional)
        Any extra arguments for the function.

    """

    def step(self, t, w, dt):
        """ Step forward the vector w by the given timestep.

            Parameters
            ----------
            dt : numeric
                The timestep to move forward.
        """

        # Runge-Kutta Fehlberg formulas (see: Numerical Recipes)
        F = lambda t,w: self.F(t,w,*self._func_args)

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

    def run(self, w0, mmap=None, **time_spec):
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
            An array of times.

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
        nsteps = len(times)-1
        dt = times[1]-times[0]

        w0, ws = self._prepare_ws(w0, mmap, nsteps=nsteps)
        ndim,nparticles = w0.shape

        # Set first step to the initial conditions
        ws[:,0] = w0
        w = w0.copy()
        for ii in range(1,nsteps+1):
            w = self.step(times[ii], w, dt)
            ws[:,ii] = w

        if ws.shape[-1] == 1:
            ws = ws[...,0]
        return times, ws
