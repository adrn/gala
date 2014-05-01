# coding: utf-8

""" Utilities for nonlinear dynamics """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import scipy
from scipy import fftpack

# Project

__all__ = ['lyapunov', 'frequency_map']

# Create logger
logger = logging.getLogger(__name__)

def lyapunov(w0, integrator, dt, nsteps, d0=1e-5, nsteps_per_pullback=10, noffset=8, t1=0.):
    """ Compute the Lyapunov exponent of an orbit by integrating an orbit
        from initial conditions w0, and several nearby orbits, offset
        initially with a small separation, d0.

        Parameters
        ----------
        w0 : array_like
            Initial conditions for all phase-space coordinates.
        integrator : streamteam.Integrator
            An instantiated Integrator object. Must have a run() method.
        dt : numeric
            Timestep.
        nsteps : int
            Number of steps to run for.
        d0 : numeric (optional)
            The initial separation.
        nsteps_per_pullback : int (optional)
            Number of steps to run before re-normalizing the offset vectors.
        noffset : int (optional)
            Number of offset orbits to run.
        t1 : numeric (optional)
            Time of initial conditions. Assumed to be t=0.

    """

    w0 = np.atleast_2d(w0)

    # number of iterations
    niter = nsteps // nsteps_per_pullback
    ndim = w0.shape[1]

    # define offset vectors to start the offset orbits on
    d0_vec = np.random.uniform(size=(noffset,ndim))
    d0_vec /= np.linalg.norm(d0_vec, axis=1)[:,np.newaxis]
    d0_vec *= d0

    w_offset = w0 + d0_vec
    all_w0 = np.vstack((w0,w_offset))

    # array to store the full, main orbit
    full_w = np.zeros((nsteps+1,ndim))
    full_w[0] = w0

    # arrays to store the Lyapunov exponents and times
    LEs = np.zeros((niter,noffset))
    ts = np.zeros_like(LEs)

    time = 0.
    for i in range(1,niter+1):
        ii = i * nsteps_per_pullback

        tt,ww = integrator.run(all_w0, dt=dt, nsteps=nsteps_per_pullback, t1=time)
        time += dt*nsteps_per_pullback

        main_w = ww[-1,0][np.newaxis]
        d1 = ww[-1,1:] - main_w
        d1_mag = np.linalg.norm(d1, axis=1)

        LEs[i-1] = np.log(d1_mag/d0)
        ts[i-1] = time

        w_offset = ww[-1,0] + d0 * d1 / d1_mag[:,np.newaxis]
        all_w0 = np.vstack((ww[-1,0],w_offset))

        full_w[(i-1)*nsteps_per_pullback+1:ii+1] = ww[1:,0]

    LEs = np.array([LEs[:ii].sum(axis=0)/ts[ii-1] for ii in range(1,niter)])

    return LEs, full_w

def frequency_map(w0, integrator, dt, nsteps, t1=0.):
    """ TODO...

        Parameters
        ----------
        w0 : array_like
            Initial conditions for all phase-space coordinates.
        integrator : streamteam.Integrator
            An instantiated Integrator object. Must have a run() method.
        dt : numeric (optional)
            Timestep.
        nsteps : int (optional)
            Number of steps to run for.
        t1 : numeric (optional)
            Time of initial conditions. Assumed to be t=0.

    """

    w0 = np.atleast_2d(w0)
    ndim = w0.shape[1]

    # compute the orbit
    ts,ws = integrator.run(w0, dt=dt, nsteps=nsteps)

    ffts = np.zeros((nsteps+1,ndim))
    freqs = np.zeros((nsteps+1,ndim))
    for ii in range(ndim):
        ffts[:,ii] = np.abs(scipy.fft(ws[:,0,ii]))
        freqs[:,ii] = fftpack.fftfreq(nsteps+1, dt)

    return freqs, ffts