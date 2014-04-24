# coding: utf-8

""" Utilities for nonlinear dynamics """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np

# Project
# ...

__all__ = ['lyapunov']

# Create logger
logger = logging.getLogger(__name__)

def lyapunov(w0, integrator, dt, nsteps, d0=1e-5, nsteps_per_pullback=10):
    """ Compute the Lyapunov exponent of an orbit by integrating an orbit
        from initial conditions w0, and an orbit with a small offset, d0.

        Parameters
        ----------
        w0 : array_like
            Initial conditions for all phase-space coordinates.
        integrator : streamteam.Integrator
        dt : numeric
        nsteps : int
        d0 : numeric
        nsteps_per_pullback : int

    """

    niter = nsteps // nsteps_per_pullback
    ndim = w0.size

    # define an offset vector to start the offset orbit on
    d0_vec = np.zeros_like(w0)
    d0_vec[0] = d0

    w_offset = w0 + d0_vec
    w_i = np.vstack((w0,w_offset))

    full_w = np.zeros((nsteps+1,ndim))
    full_w[0] = w0

    LEs = np.zeros(niter)
    ts = np.zeros_like(LEs)
    time = 0.
    for i in range(1,niter+1):
        ii = i * nsteps_per_pullback

        tt,ww = integrator.run(w_i, dt=dt, nsteps=nsteps_per_pullback)
        time += tt[-1]

        main_w = ww[-1,0]
        d1 = ww[-1,1] - main_w
        d1_mag = np.linalg.norm(d1)
        LEs[i-1] = np.log(d1_mag/d0)
        ts[i-1] = time

        w_offset = ww[-1,0] + d0 * d1 / d1_mag
        w_i = np.vstack((ww[-1,0],w_offset))

        full_w[(i-1)*nsteps_per_pullback+1:ii+1] = ww[1:,0]

    LEs = np.array([LEs[:ii].sum()/ts[ii-1] for ii in range(1,niter)])

    return LEs, full_w