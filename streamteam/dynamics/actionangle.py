# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
# ...

__all__ = ['']

def L(w):
    """
    Compute the angular momentum vector of phase-space point(s), `w`

    Parameters
    ----------
    w : array_like
        Array of phase-space positions.
    """
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
    Ls = L(w)

    # if only 2D, add another empty axis
    if w.ndim == 2:
        ntimesteps,ndim = w.shape
        w = w.reshape(ntimesteps*norbits,1,ndim)

    ntimes,norbits,ndim = w.shape

    # initial angular momentum
    L0 = Ls[0]

    # see if at any timestep the sign has changed
    loop = np.ones((norbits,3))
    for ii in range(3):
        ix = np.any(np.sign(L0[...,ii]) != np.sign(Ls[1:,...,ii]), axis=0)
        loop[ix,ii] = 0

    return loop.astype(int)

def flip_coords(w, loop_bit):
    """
    Align circulation with z-axis.

    Parameters
    ----------
    w : array_like
        Array of phase-space positions.
    loop_bit : array_like
        Array of bits that specify axis about which the orbit circulates.
        See docstring for `classify_orbit()`.
    """
    ix = loop_bit[:,0] == 1
    w[:,ix,:3] = w[:,ix,2::-1] # index magic to flip positions
    w[:,ix,3:] = w[:,ix,:2:-1] # index magic to flip velocities
    return w

def find_actions(t, w, N_max=8):
    """
    Find approximate actions and angles for samples of a phase-space orbit,
    `w`, at times `t`. Uses toy potentials with known, analytic action-angle
    transformations to approximate the true coordinates as a Fourier sum. Uses
    the formalism presented in Sanders & Binney (2014). This code is adapted
    from Jason Sanders' `genfunc <https://github.com/jlsanders/genfunc>`_

    Parameters
    ----------
    t : array_like
        Array of times with shape (ntimes,).
    w : array_like
        Phase-space orbits at times, `t`. Should have shape (ntimes,norbits,6).
    N_max : int
        Maximum integer Fourier mode vector length, |n|.
    """

    # first determine orbit
