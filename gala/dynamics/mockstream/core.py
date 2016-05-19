# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.units as u
import numpy as np

# Project
from .. import CartesianPhaseSpacePosition, Orbit
from ...potential import CPotentialBase
from ...integrate import DOPRI853Integrator, LeapfrogIntegrator
from ._mockstream import _mock_stream_dop853#, _mock_stream_leapfrog

__all__ = ['mock_stream', 'streakline_stream', 'fardal_stream', 'dissolved_fardal_stream']

def mock_stream(potential, prog_orbit, prog_mass, k_mean, k_disp,
                release_every=1, Integrator=DOPRI853Integrator, Integrator_kwargs=dict()):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    Parameters
    ----------
    potential : `~gala.potential.PotentialBase`
        The gravitational potential.
    prog_orbit : `~gala.dynamics.Orbit`
        The orbit of the progenitor system.
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
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gala.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    stream : `~gala.dynamics.CartesianPhaseSpacePosition`

    """

    # ------------------------------------------------------------------------
    # Some initial checks to short-circuit if input is bad
    if Integrator not in [LeapfrogIntegrator, DOPRI853Integrator]:
        raise ValueError("Only Leapfrog and dop853 integration is supported for"
                         " generating mock streams.")

    if not isinstance(potential, CPotentialBase):
        raise ValueError("Input potential must be a CPotentialBase subclass.")

    if not isinstance(prog_orbit, Orbit):
        raise ValueError("Progenitor orbit must be an Orbit subclass.")

    k_mean = np.atleast_1d(k_mean)
    k_disp = np.atleast_1d(k_disp)

    if k_mean.ndim > 1:
        assert k_mean.shape[0] == prog_orbit.t.size
        assert k_disp.shape[0] == prog_orbit.t.size

    # ------------------------------------------------------------------------

    if prog_orbit.t[1] < prog_orbit.t[0]:
        raise ValueError("Progenitor orbit goes backwards in time. Streams can only "
                         "be generated on orbits that run forwards. Hint: you can "
                         "reverse the orbit with prog_orbit[::-1], but make sure the array"
                         "of k_mean values is ordered correctly.")

    c_w = np.squeeze(prog_orbit.w(potential.units)).T # transpose for Cython funcs
    prog_w = np.ascontiguousarray(c_w)
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

    return CartesianPhaseSpacePosition.from_w(w=stream_w.T, units=potential.units)

def streakline_stream(potential, prog_orbit, prog_mass, release_every=1,
                      Integrator=DOPRI853Integrator, Integrator_kwargs=dict()):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the Streakline method from Kuepper et al. (2012).

    Parameters
    ----------
    potential : `~gala.potential.PotentialBase`
        The gravitational potential.
    prog_orbit : `~gala.dynamics.Orbit`
            The orbit of the progenitor system.
    prog_mass : numeric, array_like
        A single mass or an array of masses if the progenitor mass evolves
        with time.
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gala.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    stream : `~gala.dynamics.CartesianPhaseSpacePosition`

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

    return mock_stream(potential=potential, prog_orbit=prog_orbit, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs)

def fardal_stream(potential, prog_orbit, prog_mass, release_every=1,
                  Integrator=DOPRI853Integrator, Integrator_kwargs=dict()):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the prescription from Fardal et al. (2015).

    Parameters
    ----------
    potential : `~gala.potential.PotentialBase`
        The gravitational potential.
    prog_orbit : `~gala.dynamics.Orbit`
            The orbit of the progenitor system.
    prog_mass : numeric, array_like
        A single mass or an array of masses if the progenitor mass evolves
        with time.
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gala.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    stream : `~gala.dynamics.CartesianPhaseSpacePosition`

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

    return mock_stream(potential=potential, prog_orbit=prog_orbit, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs)

def dissolved_fardal_stream(potential, prog_orbit, prog_mass, t_disrupt,
                            release_every=1, Integrator=DOPRI853Integrator, Integrator_kwargs=dict()):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the prescription from Fardal et al. (2015), but at a specified
    time the progenitor completely dissolves and the radial offset of the
    tidal radius is reduced to 0 at fixed dispersion.

    Parameters
    ----------
    potential : `~gala.potential.PotentialBase`
        The gravitational potential.
    prog_orbit : `~gala.dynamics.Orbit`
            The orbit of the progenitor system.
    prog_mass : numeric, array_like
        A single mass or an array of masses if the progenitor mass evolves
        with time.
    t_disrupt : numeric
        The time that the progenitor completely disrupts.
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gala.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    stream : `~gala.dynamics.CartesianPhaseSpacePosition`

    """

    # the time index closest to when the disruption happens
    t = prog_orbit.t
    disrupt_ix = np.abs(t - t_disrupt).argmin()

    k_mean = np.zeros((t.size,6))
    k_disp = np.zeros((t.size,6))

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

    return mock_stream(potential=potential, prog_orbit=prog_orbit, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs)
