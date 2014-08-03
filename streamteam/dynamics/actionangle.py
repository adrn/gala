# coding: utf-8

from __future__ import division, print_function

"""
Utilities for estimating actions and angles for an arbitrary orbit in an
arbitrary potential.
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import time

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u
from scipy.linalg import solve
from scipy.optimize import leastsq

# Project
from .core import angular_momentum, classify_orbit
from ..potential import HarmonicOscillatorPotential, IsochronePotential

__all__ = ['cross_validate_actions', 'find_actions', 'generate_n_vectors', \
           'fit_isochrone', 'fit_harmonic_oscillator']

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

def generate_n_vectors(N_max, dx=1, dy=1, dz=1):
    """
    Generate integer vectors with |n| < N_max in just half of the three-
    dimensional lattice. If the set N = {(i,j,k)} defines the lattice,
    we restrict to the cases such that (k > 0), (k = 0, j > 0), and
    (k = 0, j = 0, i > 0).

    Parameters
    ----------
    N_max : int
        Maximum norm of the integer vector.
    dx : int
        Step size in x direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dy : int
        Step size in y direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dz : int
        Step size in z direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.

    """
    vecs = np.meshgrid(np.arange(-N_max, N_max+1, dx),
                       np.arange(-N_max, N_max+1, dy),
                       np.arange(-N_max, N_max+1, dz))
    vecs = np.vstack(map(np.ravel,vecs)).T
    vecs = vecs[np.linalg.norm(vecs,axis=1) <= N_max]
    ix = ((vecs[:,2] > 0) | ((vecs[:,2] == 0) & (vecs[:,1] > 0)) | ((vecs[:,2] == 0) & (vecs[:,1] == 0) & (vecs[:,0] > 0)))
    vecs = vecs[ix]
    return np.array(sorted(vecs, key=lambda x: (x[0],x[1],x[2])))

def unroll_angles(angles, sign=1.):
    """
    Unrolls the angles, `angles`, so they increase continuously instead of
    wrapping at 2π.

    Parameters
    ----------
    angles : array_like
        Array of angles, (ntimes,3).
    sign : numeric (optional)
        Vector that defines direction of circulation about the axes.
    """
    n = np.array([0,0,0])
    P = np.zeros_like(angles)
    P[0] = angles[0]

    n = np.cumsum(((angles[1:] - angles[:-1] + 0.5*sign*np.pi)*sign < 0) * 2.*np.pi, axis=0)
    P[1:] = angles[1:] + sign*n
    return P

def check_angle_sampling(nvecs, angles):
    """
    Returns a list of the index of elements of n which do not have adequate
    toy angle coverage. The criterion is that we must have at least one sample
    in each Nyquist box when we project the toy angles along the vector n.

    Parameters
    ----------
    nvecs : array_like
        Array of integer vectors.
    angles : array_like
        Array of angles.
    """

    checks = np.array([])
    P = np.array([])

    logger.debug("Checking modes:")
    for i,vec in enumerate(nvecs):
        N = np.linalg.norm(vec)
        X = np.dot(angles,vec)

        if(np.abs(np.max(X)-np.min(X)) < 2.*np.pi):
            logger.warning("Need a longer integration window for mode " + str(vec))
            checks = np.append(checks,vec)
            P = np.append(P,(2.*np.pi-np.abs(np.max(X)-np.min(X))))

        elif(np.abs(np.max(X)-np.min(X))/len(X) > np.pi):
            logger.warning("Need a finer sampling for mode " + str(vec))
            checks = np.append(checks,vec)
            P = np.append(P,(2.*np.pi-np.abs(np.max(X)-np.min(X))))

    return checks,P

def _action_prepare(aa, N_max, dx, dy, dz, sign=1.):
    """
    Given toy actions and angles, `aa`, compute the matrix `A` and
    vector `b` to solve for the vector of "true" actions and generating
    function values, `x` (see Equations 12-14 in Sanders & Binney (2014)).

    Parameters
    ----------
    aa : array_like
        Shape (ntimes,6) array of toy actions and angles.
    N_max : int
        Maximum norm of the integer vector.
    dx : int
        Step size in x direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dy : int
        Step size in y direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dz : int
        Step size in z direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    sign : numeric (optional)
        Vector that defines direction of circulation about the axes.
    """

    # unroll the angles so they increase continuously instead of wrap
    angles = unroll_angles(aa[:,3:], sign=sign)

    # generate integer vectors for fourier modes
    nvecs = generate_n_vectors(N_max, dx, dy, dz)

    # make sure we have enough angle coverage
    modes,P = check_angle_sampling(nvecs, angles)

    # TODO: throw out modes?
    # if(throw_out_modes):
    #     n_vectors = np.delete(n_vectors,check_each_direction(n_vectors,angs),axis=0)

    n = len(nvecs) + 3
    b = np.zeros(shape=(n, ))
    A = np.zeros(shape=(n,n))

    # top left block matrix: identity matrix summed over timesteps
    A[:3,:3] = len(aa)*np.identity(3)

    actions = aa[:,:3]
    angles = aa[:,3:]

    # top right block matrix: transpose of C_nk matrix (Eq. 12)
    C_T = 2.*nvecs.T * np.sum(np.cos(np.dot(nvecs,angles.T)), axis=-1)
    A[:3,3:] = C_T
    A[3:,:3] = C_T.T

    # lower right block matrix: C_nk dotted with C_nk^T
    cosv = np.cos(np.dot(nvecs,angles.T))
    A[3:,3:] = 4.*np.dot(nvecs,nvecs.T)*np.einsum('it,jt->ij', cosv, cosv)

    # b vector first three is just sum of toy actions
    b[:3] = np.sum(actions, axis=0)

    # rest of the vector is C dotted with actions
    b[3:] = 2*np.sum(np.dot(nvecs,actions.T)*np.cos(np.dot(nvecs,angles.T)), axis=1)

    return A,b,nvecs

def _angle_prepare(aa, t, N_max, dx, dy, dz, sign=1.):
    """
    Given toy actions and angles, `aa`, compute the matrix `A` and
    vector `b` to solve for the vector of "true" angles, frequencies, and
    generating function derivatives, `x` (see Appendix of
    Sanders & Binney (2014)).

    Parameters
    ----------
    aa : array_like
        Shape (ntimes,6) array of toy actions and angles.
    t : array_like
        Array of times.
    N_max : int
        Maximum norm of the integer vector.
    dx : int
        Step size in x direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dy : int
        Step size in y direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    dz : int
        Step size in z direction. Set to 1 for odd and even terms, set
        to 2 for just even terms.
    sign : numeric (optional)
        Vector that defines direction of circulation about the axes.
    """

    # unroll the angles so they increase continuously instead of wrap
    angles = unroll_angles(aa[:,3:], sign=sign)

    # generate integer vectors for fourier modes
    nvecs = generate_n_vectors(N_max, dx, dy, dz)

    # make sure we have enough angle coverage
    modes,P = check_angle_sampling(nvecs, angles)

    # TODO: throw out modes?
    # if(throw_out_modes):
    #     n_vectors = np.delete(n_vectors,check_each_direction(n_vectors,angs),axis=0)

    nv = len(nvecs)
    n = 3 + 3 + 3*nv # angle(0)'s, freqs, 3 derivatives of Sn

    b = np.zeros(shape=(n,))
    A = np.zeros(shape=(n,n))

    # top left block matrix: identity matrix summed over timesteps
    A[:3,:3] = len(aa)*np.identity(3)

    # identity matrices summed over times
    A[:3,3:6] = A[3:6,:3] = np.sum(t)*np.identity(3)
    A[3:6,3:6] = np.sum(t*t)*np.identity(3)

    # S1,2,3
    A[6:6+nv,0] = -2.*np.sum(np.sin(np.dot(nvecs,angles.T)),axis=1)
    A[6+nv:6+2*nv,1] = A[6:6+nv,0]
    A[6+2*nv:6+3*nv,2] = A[6:6+nv,0]

    # t*S1,2,3
    A[6:6+nv,3] = -2.*np.sum(t[None,:]*np.sin(np.dot(nvecs,angles.T)),axis=1)
    A[6+nv:6+2*nv,4] = A[6:6+nv,3]
    A[6+2*nv:6+3*nv,5] = A[6:6+nv,3]

    # lower right block structure: S dot S^T
    sinv = np.sin(np.dot(nvecs,angles.T))
    SdotST = np.einsum('it,jt->ij', sinv, sinv)
    A[6:6+nv,6:6+nv] = A[6+nv:6+2*nv,6+nv:6+2*nv] = \
        A[6+2*nv:6+3*nv,6+2*nv:6+3*nv] = 4*SdotST

    # top rectangle
    A[:6,:] = A[:,:6].T

    b[:3] = np.sum(angles, axis=0)
    b[3:6] = np.sum(t[:,None]*angles, axis=0)
    b[6:6+nv] = -2.*np.sum(angles[:,0]*np.sin(np.dot(nvecs,angles.T)), axis=1)
    b[6+nv:6+2*nv] = -2.*np.sum(angles[:,1]*np.sin(np.dot(nvecs,angles.T)), axis=1)
    b[6+2*nv:6+3*nv] = -2.*np.sum(angles[:,2]*np.sin(np.dot(nvecs,angles.T)), axis=1)

    return A,b,nvecs

def fit_isochrone(w, usys, m0=2E11, b0=1.):
    """
    Fit the toy Isochrone potential to the sum of the energy residuals.

    Parameters
    ----------
    w : array_like
        Array of phase-space positions.
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units. For example,
        (u.kpc, u.Myr, u.Msun).
    m0 : numeric (optional)
        Initial mass guess.
    b0 : numeric (optional)
        Initial b guess.
    """
    # find best toy potential parameters
    potential = IsochronePotential(m=1E10, b=10., usys=usys)
    def f(p,w):
        logm,b = p
        potential.parameters['m'] = np.exp(logm)
        potential.parameters['b'] = b
        H = potential.energy(w[...,:3], w[...,3:])
        return np.squeeze(H - np.median(H))

    logm0 = np.log(m0)
    p,ier = leastsq(f, np.array([logm0, b0]), args=(w,))
    if ier < 1 or ier > 4:
        raise ValueError("Failed to fit toy potential to orbit.")

    logm,b = np.abs(p)
    m = np.exp(logm)
    return m,b

def fit_harmonic_oscillator(w, usys, omega=[1.,1.,1.]):
    """
    Fit the toy Harmonic Oscillator potential to the sum of the
    energy residuals.

    Parameters
    ----------
    w : array_like
        Array of phase-space positions.
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units. For example,
        (u.kpc, u.Myr, u.Msun).
    omega : array_like (optional)
        Initial frequency guess.
    """
    # find best toy potential parameters
    potential = HarmonicOscillatorPotential(omega=[1.,1.,1.])
    def f(omega,w):
        potential.parameters['omega'] = omega
        H = potential.energy(w[...,:3], w[...,3:])
        return np.squeeze(H - np.median(H))

    p,ier = leastsq(f, np.array(omega), args=(w,))
    if ier < 1 or ier > 4:
        raise ValueError("Failed to fit toy potential to orbit.")

    best_omega = np.abs(p)
    return best_omega

def find_actions(t, w, N_max, usys, return_Sn=False, force_harmonic_oscillator=False):
    """
    Find approximate actions and angles for samples of a phase-space orbit,
    `w`, at times `t`. Uses toy potentials with known, analytic action-angle
    transformations to approximate the true coordinates as a Fourier sum.

    This code is adapted from Jason Sanders'
    `genfunc <https://github.com/jlsanders/genfunc>`_

    Parameters
    ----------
    t : array_like
        Array of times with shape (ntimes,).
    w : array_like
        Phase-space orbit at times, `t`. Should have shape (ntimes,6).
    N_max : int
        Maximum integer Fourier mode vector length, |n|.
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units. For example,
        (u.kpc, u.Myr, u.Msun).
    return_Sn : bool (optional)
        Return the Sn and dSn/dJ's. Default is False.
    force_harmonic_oscillator : bool (optional)
        Force using the harmonic oscillator potential as the toy potential.
    """

    if w.ndim > 2:
        raise ValueError("w must be a single orbit")

    orbit_class = classify_orbit(w)
    if np.any(orbit_class == 1) and not force_harmonic_oscillator: # loop orbit
        logger.debug("===== Loop orbit =====")
        logger.debug("Using isochrone toy potential")

        m,b = fit_isochrone(w, usys=usys)
        potential = IsochronePotential(m=m, b=b, usys=usys)
        logger.debug("Best m={}, b={}".format(m, b))

        dxyz = (1,2,2)
        circ = np.sign(w[0,0]*w[0,4]-w[0,1]*w[0,3])
        sign = np.array([1.,circ,1.])

    else: # box orbit
        logger.debug("===== Box orbit =====")
        logger.debug("Using triaxial harmonic oscillator toy potential")

        omega = fit_harmonic_oscillator(w, usys=usys)
        potential = HarmonicOscillatorPotential(omega=omega)
        logger.debug("Best omegas ({})".format(omega))

        dxyz = (2,2,2)
        sign = 1.

    # Now find toy actions and angles
    aa = np.hstack(potential.action_angle(w[...,:3], w[...,3:]))
    if np.any(np.isnan(aa)):
        ix = ~np.any(np.isnan(aa),axis=1)
        aa = aa[ix]
        t = t[ix]
        logger.warning("NaN value in toy actions or angles!")
        if sum(ix) > 1:
            raise ValueError("Too many NaN value in toy actions or angles!")

    t1 = time.time()
    A,b,nvecs = _action_prepare(aa, N_max, dx=dxyz[0], dy=dxyz[1], dz=dxyz[2])
    actions = np.array(solve(A,b))
    logger.debug("Action solution found for N_max={}, size {} symmetric"
                 " matrix in {} seconds"\
                 .format(N_max,len(actions),time.time()-t1))

    t1 = time.time()
    A,b,nvecs = _angle_prepare(aa, t, N_max, dx=dxyz[0], dy=dxyz[1], dz=dxyz[2], sign=sign)
    angles = np.array(solve(A,b))
    logger.debug("Angle solution found for N_max={}, size {} symmetric"
                 " matrix in {} seconds"\
                 .format(N_max,len(angles),time.time()-t1))

    # Just some checks
    if len(angles) > len(aa):
        logger.warning("More unknowns than equations!")

    J = actions[:3] * sign
    theta = angles[:3]
    freq = angles[3:6] * sign

    if return_Sn:
        return J, theta, freq, actions[3:], angles[6:], nvecs
    else:
        return J, theta, freq

def cross_validate_actions(t, w, N_max, usys, nbins=10, skip_failures=False):
    """
    Compute actions along windows of time of an orbit, `w`, to make sure
    the solutions are stable.

    The integration time must be long enough that it can be broken into `nbins`
    overlapping samples.

    Parameters
    ----------
    t : array_like
        Array of times with shape (ntimes,).
    w : array_like
        Phase-space orbit at times, `t`. Should have shape (ntimes,6).
    N_max : int
        Maximum integer Fourier mode vector length, |n|.
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units. For example,
        (u.kpc, u.Myr, u.Msun).
    nbins : int (optional)
        Number of bins to split the input orbit into.
    skip_failures : bool (optional)
        Skip any individual failure of `find_actions()`, but keep trying
        on other bins.
    """
    t_split = np.array_split(t,nbins)
    w_split = np.array_split(w,nbins)

    all_actions, all_angles, all_freqs = [],[],[]
    for tt,ww in zip(t_split,w_split):
        if skip_failures:
            try:
                actions,angles,freqs = find_actions(tt, ww, N_max, usys, return_Sn=False)
            except:
                logger.debug("Skipping failure...")
                continue
        else:
            actions,angles,freqs = find_actions(tt, ww, N_max, usys, return_Sn=False)

        all_actions.append(actions)
        all_angles.append(angles)
        all_freqs.append(freqs)

    if len(all_actions) == 0:
        raise ValueError("No action calculations were successful!")

    all_actions, all_angles, all_freqs = map(np.array, [all_actions, all_angles, all_freqs])

    return all_actions, all_angles, all_freqs
