# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np
from scipy.signal import argrelmin

# Project
from . import CartesianPhaseSpacePosition, CartesianOrbit

__all__ = ['fast_lyapunov_max', 'lyapunov_max', 'surface_of_section']

def fast_lyapunov_max(w0, potential, dt, n_steps, d0=1e-5,
                      n_steps_per_pullback=10, noffset_orbits=2, t1=0.,
                      atol=1E-10, rtol=1E-10, nmax=0, return_orbit=True):
    """
    Compute the maximum Lyapunov exponent using a C-implemented estimator
    that uses the DOPRI853 integrator.

    Parameters
    ----------
    w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
        Initial conditions.
    dt : numeric
        Timestep.
    n_steps : int
        Number of steps to run for.
    d0 : numeric (optional)
        The initial separation.
    n_steps_per_pullback : int (optional)
        Number of steps to run before re-normalizing the offset vectors.
    noffset_orbits : int (optional)
        Number of offset orbits to run.
    t1 : numeric (optional)
        Time of initial conditions. Assumed to be t=0.
    return_orbit : bool (optional)
        Store the full orbit for the parent and all offset orbits.

    Returns
    -------
    LEs : :class:`~astropy.units.Quantity`
        Lyapunov exponents calculated from each offset / deviation orbit.
    orbit : `~gala.dynamics.CartesianOrbit` (optional)

    """

    from .lyapunov import dop853_lyapunov_max, dop853_lyapunov_max_dont_save

    if not hasattr(potential, 'c_instance'):
        raise TypeError("Input potential must be a CPotential subclass.")

    if not isinstance(w0, CartesianPhaseSpacePosition):
        w0 = np.asarray(w0)
        ndim = w0.shape[0]//2
        w0 = CartesianPhaseSpacePosition(pos=w0[:ndim],
                                         vel=w0[ndim:])

    _w0 = np.squeeze(w0.w(potential.units))
    if _w0.ndim > 1:
        raise ValueError("Can only compute fast Lyapunov exponent for a single orbit.")

    if return_orbit:
        t,w,l = dop853_lyapunov_max(potential.c_instance, _w0,
                                    dt, n_steps+1, t1,
                                    d0, n_steps_per_pullback, noffset_orbits,
                                    atol, rtol, nmax)
        w = np.rollaxis(w, -1)

        try:
            tunit = potential.units['time']
        except (TypeError, AttributeError):
            tunit = u.dimensionless_unscaled

        orbit = CartesianOrbit.from_w(w=w, units=potential.units, t=t*tunit, potential=potential)
        return l/tunit, orbit
    else:
        l = dop853_lyapunov_max_dont_save(potential.c_instance, _w0,
                                          dt, n_steps+1, t1,
                                          d0, n_steps_per_pullback, noffset_orbits,
                                          atol, rtol, nmax)

        try:
            tunit = potential.units['time']
        except (TypeError, AttributeError):
            tunit = u.dimensionless_unscaled

        return l/tunit

def lyapunov_max(w0, integrator, dt, n_steps, d0=1e-5, n_steps_per_pullback=10,
                 noffset_orbits=8, t1=0., units=None):
    """

    Compute the maximum Lyapunov exponent of an orbit by integrating many
    nearby orbits (``noffset``) separated with isotropically distributed
    directions but the same initial deviation length, ``d0``. This algorithm
    re-normalizes the offset orbits every ``n_steps_per_pullback`` steps.

    Parameters
    ----------
    w0 : `~gala.dynamics.PhaseSpacePosition`, array_like
        Initial conditions.
    integrator : `~gala.integrate.Integrator`
        An instantiated `~gala.integrate.Integrator` object. Must have a run() method.
    dt : numeric
        Timestep.
    n_steps : int
        Number of steps to run for.
    d0 : numeric (optional)
        The initial separation.
    n_steps_per_pullback : int (optional)
        Number of steps to run before re-normalizing the offset vectors.
    noffset_orbits : int (optional)
        Number of offset orbits to run.
    t1 : numeric (optional)
        Time of initial conditions. Assumed to be t=0.
    units : `~gala.units.UnitSystem` (optional)
        If passing in an array (not a `~gala.dynamics.PhaseSpacePosition`),
        you must specify a unit system.

    Returns
    -------
    LEs : :class:`~astropy.units.Quantity`
        Lyapunov exponents calculated from each offset / deviation orbit.
    orbit : `~gala.dynamics.CartesianOrbit`
    """

    if units is not None:
        pos_unit = units['length']
        vel_unit = units['length']/units['time']
    else:
        pos_unit = u.dimensionless_unscaled
        vel_unit = u.dimensionless_unscaled

    if not isinstance(w0, CartesianPhaseSpacePosition):
        w0 = np.asarray(w0)
        ndim = w0.shape[0]//2
        w0 = CartesianPhaseSpacePosition(pos=w0[:ndim]*pos_unit,
                                         vel=w0[ndim:]*vel_unit)

    _w0 = w0.w(units)
    ndim = 2*w0.ndim

    # number of iterations
    niter = n_steps // n_steps_per_pullback

    # define offset vectors to start the offset orbits on
    d0_vec = np.random.uniform(size=(ndim,noffset_orbits))
    d0_vec /= np.linalg.norm(d0_vec, axis=0)[np.newaxis]
    d0_vec *= d0

    w_offset = _w0 + d0_vec
    all_w0 = np.hstack((_w0,w_offset))

    # array to store the full, main orbit
    full_w = np.zeros((ndim,n_steps+1,noffset_orbits+1))
    full_w[:,0] = all_w0
    full_ts = np.zeros((n_steps+1,))
    full_ts[0] = t1

    # arrays to store the Lyapunov exponents and times
    LEs = np.zeros((niter,noffset_orbits))
    ts = np.zeros_like(LEs)
    time = t1
    for i in range(1,niter+1):
        ii = i * n_steps_per_pullback

        orbit = integrator.run(all_w0, dt=dt, n_steps=n_steps_per_pullback, t1=time)
        tt = orbit.t.value
        ww = orbit.w(units)
        time += dt*n_steps_per_pullback

        main_w = ww[:,-1,0:1]
        d1 = ww[:,-1,1:] - main_w
        d1_mag = np.linalg.norm(d1, axis=0)

        LEs[i-1] = np.log(d1_mag/d0)
        ts[i-1] = time

        w_offset = ww[:,-1,0:1] + d0 * d1 / d1_mag[np.newaxis]
        all_w0 = np.hstack((ww[:,-1,0:1],w_offset))

        full_w[:,(i-1)*n_steps_per_pullback+1:ii+1] = ww[:,1:]
        full_ts[(i-1)*n_steps_per_pullback+1:ii+1] = tt[1:]

    LEs = np.array([LEs[:ii].sum(axis=0)/ts[ii-1] for ii in range(1,niter)])

    try:
        t_unit = units['time']
    except (TypeError, AttributeError):
        t_unit = u.dimensionless_unscaled

    orbit = CartesianOrbit.from_w(w=full_w, units=units, t=full_ts*t_unit)
    return LEs/t_unit, orbit

def surface_of_section(orbit, plane_ix, interpolate=False):
    """
    Generate and return a surface of section from the given orbit.

    .. warning::

        This is an experimental function and the API may change.

    Parameters
    ----------
    orbit : `~gala.dynamics.CartesianOrbit`
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

    Returns
    -------

    Examples
    --------
    If your orbit of interest is a tube orbit, it probably conserves (at
    least approximately) some equivalent to angular momentum in the direction
    of the circulation axis. Therefore, a surface of section in R-z should
    be instructive for classifying these orbits. TODO...show how to convert
    an orbit to Cylindrical..etc...

    """

    w = orbit.w()
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
