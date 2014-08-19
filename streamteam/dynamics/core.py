# coding: utf-8

""" General dynamics utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project

__all__ = ['angular_momentum', 'classify_orbit']

def angular_momentum(w):
    """
    Compute the angular momentum vector(s) of phase-space point(s), `w`.

    Parameters
    ----------
    w : array_like
        Array of phase-space positions. The last axis (`axis=-1`) is assumed
        to be the phase-space dimension so that the phase-space dimensionality
        is `w.shape[-1]`.
    """
    w = np.atleast_1d(w)
    ndim = w.shape[-1]
    return np.cross(w[...,:ndim//2], w[...,ndim//2:])

def classify_orbit(w):
    """
    Determine whether an orbit is a Box or Loop orbit by figuring out
    whether there is a change of sign of the angular momentum about an
    axis. Returns an array with 3 integers per phase-space point, such
    that:

    - Box = [0,0,0]
    - Short axis Loop = [0,0,1]
    - Long axis Loop = [1,0,0]

    Parameters
    ----------
    w : array_like
        Array of phase-space positions.

    """
    # get angular momenta
    Ls = angular_momentum(w)

    # if only 2D, add another empty axis
    if w.ndim == 2:
        ntimesteps,ndim = w.shape
        w = w.reshape(ntimesteps,1,ndim)

    ntimes,norbits,ndim = w.shape

    # initial angular momentum
    L0 = Ls[0]

    # see if at any timestep the sign has changed
    loop = np.ones((norbits,3))
    for ii in range(3):
        cnd = (np.sign(L0[...,ii]) != np.sign(Ls[1:,...,ii])) | \
              (np.abs(Ls[1:,...,ii]) < 1E-14)

        ix = np.atleast_1d(np.any(cnd, axis=0))
        loop[ix,ii] = 0

    return loop.astype(int)