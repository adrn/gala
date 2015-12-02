# coding: utf-8

""" Utilities for nonlinear dynamics """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from scipy.signal import argrelmin

# Project
from ..util import atleast_2d

__all__ = ['fast_lyapunov_max', 'lyapunov_max', 'surface_of_section']

def fast_lyapunov_max(w0, potential, dt, nsteps, d0=1e-5,
                      nsteps_per_pullback=10, noffset_orbits=2, t0=0.,
                      atol=1E-9, rtol=1E-9, nmax=0):
    """
    Compute the maximum Lyapunov exponent using a C-implemented estimator
    that uses the DOPRI853 integrator.

    Parameters
    ----------
    TODO:

    Returns
    -------
    LEs : :class:`numpy.ndarray`
        Lyapunov exponents calculated from each offset / deviation orbit.
    ts : :class:`numpy.ndarray`
        Array of times from integrating main orbit.
    ws : :class:`numpy.ndarray`
        All orbits -- main / parent orbit is index 0, all others are the
        full orbits of the deviations. TODO: right now, only returns parent
        orbit.

    """

    from .lyapunov import dop853_lyapunov_max

    if not hasattr(potential, 'c_instance'):
        raise TypeError("Input potential must be a CPotential subclass.")

    t,w,l = dop853_lyapunov_max(potential.c_instance, w0,
                                dt, nsteps+1, t0,
                                d0, nsteps_per_pullback, noffset_orbits,
                                atol, rtol, nmax)
    w = np.rollaxis(w, -1)
    return l,t,w

def lyapunov_max(w0, integrator, dt, nsteps, d0=1e-5, nsteps_per_pullback=10,
                 noffset_orbits=8, t1=0.):
    """

    Compute the maximum Lyapunov exponent of an orbit by integrating many
    nearby orbits (``noffset``) separated with isotropically distributed
    directions but the same initial deviation length, ``d0``. This algorithm
    re-normalizes the offset orbits every ``nsteps_per_pullback`` steps.

    Parameters
    ----------
    w0 : array_like
        Initial conditions for all phase-space coordinates.
    integrator : gary.Integrator
        An instantiated Integrator object. Must have a run() method.
    dt : numeric
        Timestep.
    nsteps : int
        Number of steps to run for.
    d0 : numeric (optional)
        The initial separation.
    nsteps_per_pullback : int (optional)
        Number of steps to run before re-normalizing the offset vectors.
    noffset_orbits : int (optional)
        Number of offset orbits to run.
    t1 : numeric (optional)
        Time of initial conditions. Assumed to be t=0.

    Returns
    -------
    LEs : :class:`numpy.ndarray`
        Lyapunov exponents calculated from each offset / deviation orbit.
    ts : :class:`numpy.ndarray`
        Array of times from integrating main orbit.
    ws : :class:`numpy.ndarray`
        All orbits -- main / parent orbit is index 0, all others are the
        full orbits of the deviations. TODO: right now, only returns parent
        orbit.
    """

    w0 = atleast_2d(w0, insert_axis=1)

    # number of iterations
    niter = nsteps // nsteps_per_pullback
    ndim = w0.shape[0]

    # define offset vectors to start the offset orbits on
    d0_vec = np.random.uniform(size=(ndim,noffset_orbits))
    d0_vec /= np.linalg.norm(d0_vec, axis=0)[np.newaxis]
    d0_vec *= d0

    w_offset = w0 + d0_vec
    all_w0 = np.hstack((w0,w_offset))

    # array to store the full, main orbit
    full_w = np.zeros((ndim,nsteps+1,noffset_orbits+1))
    full_w[:,0] = all_w0
    full_ts = np.zeros((nsteps+1,))
    full_ts[0] = t1

    # arrays to store the Lyapunov exponents and times
    LEs = np.zeros((niter,noffset_orbits))
    ts = np.zeros_like(LEs)
    time = t1
    for i in range(1,niter+1):
        ii = i * nsteps_per_pullback

        tt,ww = integrator.run(all_w0, dt=dt, nsteps=nsteps_per_pullback, t1=time)
        time += dt*nsteps_per_pullback

        main_w = ww[:,-1,0:1]
        d1 = ww[:,-1,1:] - main_w
        d1_mag = np.linalg.norm(d1, axis=0)

        LEs[i-1] = np.log(d1_mag/d0)
        ts[i-1] = time

        w_offset = ww[:,-1,0:1] + d0 * d1 / d1_mag[np.newaxis]
        all_w0 = np.hstack((ww[:,-1,0:1],w_offset))

        full_w[:,(i-1)*nsteps_per_pullback+1:ii+1] = ww[:,1:]
        full_ts[(i-1)*nsteps_per_pullback+1:ii+1] = tt[1:]

    LEs = np.array([LEs[:ii].sum(axis=0)/ts[ii-1] for ii in range(1,niter)])

    return LEs, full_ts, full_w

def surface_of_section(w, plane_ix, interpolate=False):
    """
    Generate and return a surface of section from the given orbit.

    .. warning::

        This is an experimental function and the API may change.

    Parameters
    ----------
    w : array_like
        Array of orbits. The phase-space dimensionality is assumed to be
        the size of ``axis=-1``.
    plane_ix : int
        Integer that represents the coordinate to record crossings in. For
        example, for a 2D Hamiltonian where you want to make a SoS in
        :math:`y-p_y`, you would specify ``plane_ix=0`` (crossing the
        :math:`x` axis), and this will only record crossings for which
        :math:`p_x>0`.
    interpolate : bool (optional)
        Whether or not to interpolate on to the plane of interest. This
        makes it much slower, but will work for orbits with a coarser
        sampling.

    Examples
    --------
    If your orbit of interest is a tube orbit, it probably conserves (at
    least approximately) some equivalent to angular momentum in the direction
    of the circulation axis. Therefore, a surface of section in R-z should
    be instructive for classifying these orbits. TODO...show how to convert
    an orbit to Cylindrical..etc...

    Returns
    -------

    """

    w = atleast_2d(w, insert_axis=1)
    if w.ndim == 2:
        w = w[...,None]

    ndim,ntimes,norbits = w.shape
    H_dim = ndim // 2
    p_ix = plane_ix + H_dim

    if interpolate:
        raise NotImplementedError("Not yet implemented, sorry!")

    # record position on specified plane when orbit crosses
    all_sos = np.zeros((ndim,norbits), dtype=object)
    for n in range(norbits):
        cross_ix = argrelmin(w[plane_ix,:,n]**2)[0]
        cross_ix = cross_ix[w[p_ix,cross_ix,n] > 0.]
        sos = w[:,cross_ix,n]

        for j in range(ndim):
            all_sos[j,n] = sos[j,:]

    return all_sos
