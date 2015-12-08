# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.units as u
import numpy as np

# Project
from .. import CartesianPhaseSpacePosition
from ...potential import CPotentialBase
from ...integrate import DOPRI853Integrator, LeapfrogIntegrator
from ._mockstream import _mock_stream_dop853#, _mock_stream_leapfrog

__all__ = ['mock_stream', 'streakline_stream', 'fardal_stream', 'dissolved_fardal_stream']

def mock_stream(potential, w0, prog_mass, k_mean, k_disp,
                t_f, dt=1., t_0=0., release_every=1,
                Integrator=LeapfrogIntegrator, Integrator_kwargs=dict()):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    Parameters
    ----------
    potential : `~gary.potential.PotentialBase`
        The gravitational potential.
    w0 : `~gary.dynamics.PhaseSpacePosition`, array_like
        Initial conditions.
    prog_mass : numeric, array_like
        A single mass or an array of masses if the progenitor mass evolves
        with time.
    k_mean : `numpy.ndarray`
        Array of mean :math:`k` values (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        :math:`(R,\phi,z,v_R,v_\phi,v_z)`. If 1D, assumed constant in time. If 2D, time axis
        is axis 0.
    k_disp : `numpy.ndarray`
        Array of :math:`k` value dispersions (see Fardal et al. 2015). These are used to determine
        the exact prescription for generating the mock stream. The components are for:
        :math:`(R,\phi,z,v_R,v_\phi,v_z)`. If 1D, assumed constant in time. If 2D, time axis
        is axis 0.
    t_f : numeric
        The final time for integrating.
    t_0 : numeric (optional)
        The initial time for integrating -- the time at which ``w0`` is the position.
    dt : numeric (optional)
        The time-step.
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gary.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    prog_orbit : `~gary.dynamics.CartesianOrbit`
    stream : `~gary.dynamics.CartesianPhaseSpacePosition`

    """

    # ------------------------------------------------------------------------
    # Some initial checks to short-circuit if input is bad
    if Integrator not in [LeapfrogIntegrator, DOPRI853Integrator]:
        raise ValueError("Only Leapfrog and dop853 integration is supported for"
                         " generating mock streams.")

    if not isinstance(w0, CartesianPhaseSpacePosition):
        w0 = CartesianPhaseSpacePosition.from_w(np.atleast_1d(w0), units=potential.units)

    if w0.pos.shape[1] > 1:
        raise ValueError("Only a single phase-space position is allowed.")

    if not isinstance(potential, CPotentialBase):
        raise ValueError("Input potential must be a CPotentialBase subclass.")
    # ------------------------------------------------------------------------

    # integrate the orbit of the progenitor system
    # TODO: t1 and dt should support units
    if (t_f-t_0) < 0.:
        dt = -np.abs(dt) # make sure dt is negative if integrating backwards
    else:
        dt = np.abs(dt) # make sure dt is positive if integrating forwards

    nsteps = int(round(np.abs((t_f-t_0)/dt)))
    t = np.linspace(t_0, t_f, nsteps)

    # integrate the orbit of the progenitor system
    prog_orbit = potential.integrate_orbit(w0, t=t, Integrator=Integrator, Integrator_kwargs=Integrator_kwargs)

    # if we integrated backwards, flip it around in time
    if dt < 0:
        prog_orbit = prog_orbit[::-1]

    prog_w = np.ascontiguousarray(prog_orbit.w(potential.units)[...,0].T) # transpose for Cython funcs
    prog_t = np.ascontiguousarray(prog_orbit.t.decompose(potential.units).value)

    if Integrator == LeapfrogIntegrator:
        pass

    elif Integrator == DOPRI853Integrator:
        stream_w = _mock_stream_dop853(potential.c_instance, t=prog_t, prog_w=prog_w,
                                       release_every=release_every,
                                       _k_mean=k_mean, _k_disp=k_disp, G=potential.G,
                                       _prog_mass=prog_mass,
                                       **Integrator_kwargs)

    else:
        raise RuntimeError("Should never get here...")

    return prog_orbit, CartesianPhaseSpacePosition.from_w(w=stream_w.T, units=potential.units)

def streakline_stream(potential, w0, prog_mass, t_f, dt=1., t_0=0., release_every=1,
                      Integrator=LeapfrogIntegrator, Integrator_kwargs=dict()):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the Streakline method from Kuepper et al. (2012).

    Parameters
    ----------
    potential : `~gary.potential.PotentialBase`
        The gravitational potential.
    w0 : `~gary.dynamics.PhaseSpacePosition`, array_like
        Initial conditions.
    prog_mass : numeric, array_like
        A single mass or an array of masses if the progenitor mass evolves
        with time.
    t_f : numeric
        The final time for integrating.
    t_0 : numeric (optional)
        The initial time for integrating -- the time at which ``w0`` is the position.
    dt : numeric (optional)
        The time-step.
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gary.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    prog_orbit : `~gary.dynamics.CartesianOrbit`
    stream : `~gary.dynamics.CartesianPhaseSpacePosition`

    """
    k_mean = np.zeros(6)
    k_disp = np.zeros(6)

    k_mean[0] = 1. # R
    k_disp[0] = 0.

    k_mean[1] = 0. # phi
    k_disp[1] = 0.

    k_mean[2] = 0. # z
    k_disp[2] = 0.

    k_mean[3] = 0. # vR
    k_disp[3] = 0.

    k_mean[4] = 1. # vt
    k_disp[4] = 0.

    k_mean[5] = 0. # vz
    k_disp[5] = 0.

    return mock_stream(potential=potential, w0=w0, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp,
                       t_f=t_f, dt=dt, t_0=t_0, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs)

def fardal_stream(potential, w0, prog_mass, t_f, dt=1., t_0=0., release_every=1,
                  Integrator=LeapfrogIntegrator, Integrator_kwargs=dict()):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the prescription from Fardal et al. (2015).

    Parameters
    ----------
    potential : `~gary.potential.PotentialBase`
        The gravitational potential.
    w0 : `~gary.dynamics.PhaseSpacePosition`, array_like
        Initial conditions.
    prog_mass : numeric, array_like
        A single mass or an array of masses if the progenitor mass evolves
        with time.
    t_f : numeric
        The final time for integrating.
    t_0 : numeric (optional)
        The initial time for integrating -- the time at which ``w0`` is the position.
    dt : numeric (optional)
        The time-step.
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gary.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    prog_orbit : `~gary.dynamics.CartesianOrbit`
    stream : `~gary.dynamics.CartesianPhaseSpacePosition`

    """
    k_mean = np.zeros(6)
    k_disp = np.zeros(6)

    k_mean[0] = 2. # R
    k_disp[0] = 0.5

    k_mean[1] = 0. # phi
    k_disp[1] = 0.

    k_mean[2] = 0. # z
    k_disp[2] = 0.5

    k_mean[3] = 0. # vR
    k_disp[3] = 0.

    k_mean[4] = 0.3 # vt
    k_disp[4] = 0.5

    k_mean[5] = 0. # vz
    k_disp[5] = 0.5

    return mock_stream(potential=potential, w0=w0, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp,
                       t_f=t_f, dt=dt, t_0=t_0, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs)

def dissolved_fardal_stream(potential, w0, prog_mass, t_disrupt, t_f, dt=1., t_0=0.,
                            release_every=1, Integrator=LeapfrogIntegrator, Integrator_kwargs=dict()):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the prescription from Fardal et al. (2015), but at a specified
    time the progenitor completely dissolves and the radial offset of the
    tidal radius is reduced to 0 at fixed dispersion.

    Parameters
    ----------
    potential : `~gary.potential.PotentialBase`
        The gravitational potential.
    w0 : `~gary.dynamics.PhaseSpacePosition`, array_like
        Initial conditions.
    prog_mass : numeric, array_like
        A single mass or an array of masses if the progenitor mass evolves
        with time.
    t_disrupt : numeric
        The time that the progenitor completely disrupts.
    t_f : numeric
        The final time for integrating.
    t_0 : numeric (optional)
        The initial time for integrating -- the time at which ``w0`` is the position.
    dt : numeric (optional)
        The time-step.
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gary.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    prog_orbit : `~gary.dynamics.CartesianOrbit`
    stream : `~gary.dynamics.CartesianPhaseSpacePosition`

    """

    # the time index closest to when the disruption happens
    nsteps = int(round(np.abs((t_f-t_0)/dt)))
    t = np.linspace(t_0, t_f, nsteps)
    disrupt_ix = np.abs(t - t_disrupt).argmin()

    k_mean = np.zeros((nsteps,6))
    k_disp = np.zeros((nsteps,6))

    k_mean[:,0] = 2. # R
    k_mean[disrupt_ix:,0] = 0.
    k_disp[:,0] = 0.5

    k_mean[:,1] = 0. # phi
    k_disp[:,1] = 0.

    k_mean[:,2] = 0. # z
    k_disp[:,2] = 0.5

    k_mean[:,3] = 0. # vR
    k_disp[:,3] = 0.

    k_mean[:,4] = 0.3 # vt
    k_disp[:,4] = 0.5

    k_mean[:,5] = 0. # vz
    k_disp[:,5] = 0.5

    return mock_stream(potential=potential, w0=w0, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp,
                       t_f=t_f, dt=dt, t_0=t_0, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs)
