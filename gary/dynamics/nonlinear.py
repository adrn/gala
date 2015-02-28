# coding: utf-8

""" Utilities for nonlinear dynamics """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging

# Third-party
import numpy as np

# Project
from ..util import gram_schmidt

__all__ = ['lyapunov_spectrum', 'fast_lyapunov_max', 'lyapunov_max']

# Create logger
logger = logging.getLogger(__name__)

def lyapunov_spectrum(w0, integrator, dt, nsteps, t1=0., deviation_vecs=None):
    """ Compute the spectrum of Lyapunov exponents given equations of motions
        for small deviations.

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
        t1 : numeric (optional)
            Time of initial conditions. Assumed to be t=0.
        deviation_vecs : array_like (optional)
            Specify your own initial deviation vectors.

    """

    w0 = np.atleast_2d(w0)

    # phase-space dimensionality
    if w0.shape[0] > 1:
        raise ValueError("Initial condition vector ")
    ndim_ps = w0.shape[1]

    if deviation_vecs is None:
        # initialize (ndim_ps) deviation vectors
        A = np.zeros((ndim_ps,ndim_ps))
        for ii in range(ndim_ps):
            A[ii] = np.random.normal(0.,1.,size=ndim_ps)
            A[ii] /= np.linalg.norm(A[ii])

    else:
        raise NotImplementedError()

    all_w0 = np.zeros((ndim_ps,ndim_ps*2))
    for ii in range(ndim_ps):
        all_w0[ii] = np.append(w0,A[ii])

    # array to store the full, main orbit
    full_w = np.zeros((nsteps+1,ndim_ps))
    full_w[0] = w0
    full_ts = np.zeros((nsteps+1,))
    full_ts[0] = t1

    # arrays to store the Lyapunov exponents and times
    lyap = np.zeros((nsteps+1,ndim_ps))
    rhi = np.zeros((nsteps+1,ndim_ps))  # sum of logs

    ts = np.zeros(nsteps+1)
    time = t1
    for ii in range(1,nsteps+1):
        tt,ww = integrator.run(all_w0, dt=dt, nsteps=1, t1=time)
        time += dt

        alf = gram_schmidt(ww[-1,:,ndim_ps:])
        rhi[ii] = rhi[ii-1] + np.log(alf)
        lyap[ii] = rhi[ii]/time

        ts[ii] = time
        full_w[ii:ii+1] = ww[1:,0,:ndim_ps]
        full_ts[ii:ii+1] = tt[1:]
        all_w0 = ww[-1].copy()

    return lyap, full_ts, full_w

def fast_lyapunov_max(w0, potential, dt, nsteps, d0=1e-5,
                      nsteps_per_pullback=10, noffset_orbits=8, t1=0.):
    from ..integrate.dopri.wrap_dop853 import dop853_lyapunov

    if not hasattr(potential, 'c_instance'):
        raise TypeError("Input potential must be a CPotential subclass.")

    t,w,l = dop853_lyapunov(potential.c_instance, w0,
                            dt, nsteps, t1, 1E-8, 1E-8,
                            d0, nsteps_per_pullback, noffset_orbits)

    return l,t,w

def lyapunov_max(w0, integrator, dt, nsteps, d0=1e-5, nsteps_per_pullback=10,
                 noffset=8, t1=0.):
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
    noffset : int (optional)
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
    full_ts = np.zeros((nsteps+1,))
    full_ts[0] = t1

    # arrays to store the Lyapunov exponents and times
    LEs = np.zeros((niter,noffset))
    ts = np.zeros_like(LEs)
    time = t1
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
        full_ts[(i-1)*nsteps_per_pullback+1:ii+1] = tt[1:]

    LEs = np.array([LEs[:ii].sum(axis=0)/ts[ii-1] for ii in range(1,niter)])

    return LEs, full_ts, full_w

# def sali(w0, integrator, dt, nsteps, t1=0., deviation_vecs=None):
#     """ Compute the Smaller Alignment Index (SALI)
#         See: Skokos, Ch. 2001, J. Phys. A: Math. Gen., 34, 10029-10043

#         Parameters
#         ----------
#         w0 : array_like
#             Initial conditions for all phase-space coordinates.
#         integrator : gary.Integrator
#             An instantiated Integrator object. Must have a run() method.
#         dt : numeric
#             Timestep.
#         nsteps : int
#             Number of steps to run for.
#         d0 : numeric (optional)
#             The initial separation.
#         nsteps_per_pullback : int (optional)
#             Number of steps to run before re-normalizing the offset vectors.
#         noffset : int (optional)
#             Number of offset orbits to run.
#         t1 : numeric (optional)
#             Time of initial conditions. Assumed to be t=0.

#     """

#     w0 = np.atleast_2d(w0)

#     # phase-space dimensionality
#     if w0.shape[0] > 1:
#         raise ValueError("Initial condition vector ")
#     ndim_ps = w0.shape[1]

#     if deviation_vecs is None:
#         # initialize (ndim_ps) deviation vectors
#         A = np.zeros((ndim_ps,ndim_ps))
#         for ii in range(ndim_ps):
#             A[ii] = np.random.normal(0.,1.,size=ndim_ps)
#             A[ii] /= np.linalg.norm(A[ii])

#         vec = gram_schmidt(A)
#         A = A[:2]

#     else:
#         raise NotImplementedError()

#     all_w0 = np.zeros((2,ndim_ps*2))
#     for ii in range(2):
#         all_w0[ii] = np.append(w0,A[ii])

#     # array to store the full, main orbit
#     full_w = np.zeros((nsteps+1,ndim_ps))
#     full_w[0] = w0
#     full_ts = np.zeros((nsteps+1,))
#     full_ts[0] = t1

#     # arrays to store the sali
#     sali = np.zeros((nsteps+1,))

#     time = t1
#     for ii in range(1,nsteps+1):
#         tt,ww = integrator.run(all_w0, dt=dt, nsteps=1, t1=time)
#         time += dt

#         dm = np.sqrt(np.sum((ww[-1,0,ndim_ps:] - ww[-1,1,ndim_ps:])**2))
#         dq = np.sqrt(np.sum((ww[-1,0,ndim_ps:] + ww[-1,1,ndim_ps:])**2))
#         sali[ii] = min(dm, dq)

#         # renormalize
#         ww[-1,0,ndim_ps:] = ww[-1,0,ndim_ps:] / np.linalg.norm(ww[-1,0,ndim_ps:])
#         ww[-1,1,ndim_ps:] = ww[-1,1,ndim_ps:] / np.linalg.norm(ww[-1,1,ndim_ps:])

#         full_w[ii:ii+1] = ww[-1,0,:ndim_ps]
#         full_ts[ii:ii+1] = time
#         all_w0 = ww[-1].copy()

#     return sali, full_ts, full_w
