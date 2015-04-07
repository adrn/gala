# coding: utf-8

""" General dynamics utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
from scipy.signal import argrelmax, argrelmin

__all__ = ['angular_momentum', 'classify_orbit', 'align_circulation_with_z',
           'check_for_primes', 'peak_to_peak_period']

def angular_momentum(q, p):
    r"""
    Compute the angular momentum vector(s) of a single or array of positions
    and conjugate momenta, ``q`` and ``p``.

    .. math::

        \boldsymbol{L} = \boldsymbol{q} \times \boldsymbol{p}

    Parameters
    ----------
    q : array_like, :class:`~astropy.units.Quantity`
        Array of positions.
    p : array_like, :class:`~astropy.units.Quantity`
        Array of momenta, or, velocities if working in per-mass units.

    Returns
    -------
    L : :class:`numpy.ndarray`, :class:`~astropy.units.Quantity`
        Array of angular momentum vectors.

    Examples
    --------

        >>> import numpy as np
        >>> import astropy.units as u
        >>> q = np.array([1., 0, 0]) * u.au
        >>> p = np.array([0, 2*np.pi, 0]) * u.au/u.yr
        >>> angular_momentum(q, p)
        <Quantity [ 0.        , 0.        , 6.28318531] AU2 / yr>

    """
    try:
        return np.cross(q,p) * q.unit * p.unit
    except AttributeError:  # q,p are just array's
        return np.cross(q,p)

def classify_orbit(w):
    """
    Determine whether an orbit or series of orbits is a Box or Tube orbit by
    figuring out whether there is a change of sign of the angular momentum
    about an axis. Returns a 2D array with 3 integers per orbit point such that:

    - Box and boxlet = [0,0,0]
    - z-axis (short-axis) tube = [0,0,1]
    - x-axis (long-axis) tube = [1,0,0]

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
        the axes of the input orbit. For a single orbit, will return a
        1D array, but for multiple orbits, the shape will be (len(w), 3).

    """
    # get angular momenta
    ndim = w.shape[-1]
    Ls = angular_momentum(w[...,:ndim//2], w[...,ndim//2:])

    # if only 2D, add another empty axis
    if w.ndim == 2:
        single_orbit = True
        ntimesteps,ndim = w.shape
        w = w.reshape(ntimesteps,1,ndim)
    else:
        single_orbit = False

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

    loop = loop.astype(int)
    if single_orbit:
        return loop.reshape((ndim//2,))
    else:
        return loop

def align_circulation_with_z(w, loop_bit):
    """
    If the input orbit is a tube orbit, this function aligns the circulation
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

    orig_shape = w.shape
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

    return new_w.reshape(orig_shape)

def check_for_primes(n, max_prime=41):
    """
    Given an integer, ``n``, ensure that it doest not have large prime
    divisors, which can wreak havok for FFT's. If needed, will decrease
    the number.

    Parameters
    ----------
    n : int
        Integer number to test.

    Returns
    -------
    n2 : int
        Integer combed for large prime divisors.
    """

    m = n
    f = 2
    while (f**2 <= m):
        if m % f == 0:
            m /= f
        else:
            f += 1

    if m >= max_prime and n >= max_prime:
        n -= 1
        n = check_for_primes(n)

    return n

def peak_to_peak_period(t, f, tol=1E-2):
    """
    Estimate the period of the input time series by measuring the average
    peak-to-peak time.

    Parameters
    ----------
    t : array_like
        Time grid aligned with the input time series.
    f : array_like
        A periodic time series.

    Returns
    -------
    period : float
        The mean peak-to-peak period.
    """

    # find peaks
    max_ix = argrelmax(f, mode='wrap')[0]
    max_ix = max_ix[(max_ix != 0) & (max_ix != (len(f)-1))]

    # find troughs
    min_ix = argrelmin(f, mode='wrap')[0]
    min_ix = min_ix[(min_ix != 0) & (min_ix != (len(f)-1))]

    # neglect minor oscillations
    if abs(np.mean(f[max_ix]) - np.mean(f[min_ix])) < tol:
        return np.nan

    # compute mean peak-to-peak
    if len(max_ix) > 0:
        T_max = np.mean(t[max_ix[1:]] - t[max_ix[:-1]])
    else:
        T_max = np.nan

    # now compute mean trough-to-trough
    if len(min_ix) > 0:
        T_min = np.mean(t[min_ix[1:]] - t[min_ix[:-1]])
    else:
        T_min = np.nan

    # then take the mean of these two
    return np.mean([T_max, T_min])
