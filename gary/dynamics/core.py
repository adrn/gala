# coding: utf-8

""" General dynamics utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger

__all__ = ['angular_momentum', 'classify_orbit', 'align_circulation_with_z']

def angular_momentum(w):
    """
    Compute the angular momentum vector(s) of phase-space point(s), `w`.

    Parameters
    ----------
    w : array_like
        Array of phase-space positions. The last axis (`axis=-1`) is assumed
        to be the phase-space dimension so that the phase-space dimensionality
        is `w.shape[-1]`.

    Returns
    -------
    L : :class:`numpy.ndarray`
        Array of angular momentum vectors.
    """
    w = np.atleast_1d(w)
    ndim = w.shape[-1]
    return np.cross(w[...,:ndim//2], w[...,ndim//2:])

def classify_orbit(w):
    """
    Determine whether an orbit or series of orbits is a Box or Loop orbit by
    figuring out whether there is a change of sign of the angular momentum
    about an axis. Returns a 2D array with 3 integers per orbit point such that:

    - Box = [0,0,0]
    - Loop around z axis = [0,0,1]
    - Loop around x axis = [1,0,0]

    Parameters
    ----------
    w : array_like
        Array of phase-space positions. Accepts 2D or 3D arrays. If 2D, assumes
        this is a single orbit. If 3D, assumes that this is a collection of orbits,
        where `axis=0` is the time axis, and `axis=1` are the different orbits.

    Returns
    -------
    circulation : :class:`numpy.ndarray`
        An array that specifies whether there is circulation about any of
        the axes of the input orbit. Shape: (len(w), 3)

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

def align_circulation_with_z(w, loop_bit):
    """
    If the input orbit is a loop orbit, this function aligns the circulation
    axis with the z axis.

    Parameters
    ----------
    w : array_like
        Array of phase-space positions. Accepts 2D or 3D arrays. If 2D, assumes
        this is a single orbit so that `loop_bit` should be a 1D array. If 3D, assumes
        that this is a collection of orbits, where `axis=0` is the time axis, and
        `axis=1` are the different orbits.
    loop_bit : array_like
        Array of bits that specify the axis about which the orbit circulates.
        See the documentation for ~`gary.dynamics.classify_orbit()`.

    Returns
    -------
    new_w : :class:`~numpy.ndarray`
        A copy of the input array with circulation aligned with the z axis.
    """

    if (w.ndim-1) != loop_bit.ndim:
        raise ValueError("Shape mismatch - input orbit array should have 1 more dimension "
                         "than the input loop bit.")

    if loop_bit.ndim == 1:
        loop_bit = np.atleast_2d(loop_bit)
        w = w[:,np.newaxis]

    new_w = w.copy()
    for ix in range(len(loop_bit)):
        if loop_bit[ix,2] == 1 or np.all(loop_bit[ix] == 0):
            # already circulating about z or box orbit
            continue

        if sum(loop_bit[ix]) > 1:
            logger.warning("Circulation about x and y axes - are you sure the orbit has been "
                           "integrated for long enough?")

        if loop_bit[ix,0] == 1:
            circ = 0
        elif loop_bit[ix,1] == 1:
            circ = 1
        else:
            raise RuntimeError("Should never get here...")

        new_w[:,ix,circ] = w[:,ix,2]
        new_w[:,ix,2] = w[:,ix,circ]
        new_w[:,ix,circ+3] = w[:,ix,5]
        new_w[:,ix,5] = w[:,ix,circ+3]

    return new_w
