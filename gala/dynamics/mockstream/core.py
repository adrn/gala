# coding: utf-8

from __future__ import division, print_function


# Standard library
import warnings
import os

# Third-party
import numpy as np

# Project
from .. import PhaseSpacePosition, Orbit
from ...potential import Hamiltonian, CPotentialBase
from ...integrate import DOPRI853Integrator, LeapfrogIntegrator
from ._mockstream import _mock_stream_dop853, _mock_stream_leapfrog, _mock_stream_animate

__all__ = ['mock_stream', 'streakline_stream', 'fardal_stream', 'dissolved_fardal_stream']

def mock_stream(hamiltonian, prog_orbit, prog_mass, k_mean, k_disp,
                release_every=1, Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                snapshot_filename=None, seed=None):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    Parameters
    ----------
    hamiltonian : `~gala.potential.Hamiltonian`
        The system Hamiltonian.
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
    snapshot_filename : str (optional)
        Filename to save all incremental snapshots of particle positions and
        velocities. Warning: this can make very large files if you are not
        careful!
    seed : int (optional)
        A random number seed for initializing the particle positions.

    Returns
    -------
    stream : `~gala.dynamics.PhaseSpacePosition`

    """

    if isinstance(hamiltonian, CPotentialBase):
        warnings.warn("This function now expects a `Hamiltonian` instance instead of "
                      "a `PotentialBase` subclass instance. If you are using a "
                      "static reference frame, you just need to pass your "
                      "potential object in to the Hamiltonian constructor to use, e.g., "
                      "Hamiltonian(potential).", DeprecationWarning)

        hamiltonian = Hamiltonian(hamiltonian)

    # ------------------------------------------------------------------------
    # Some initial checks to short-circuit if input is bad
    if Integrator not in [LeapfrogIntegrator, DOPRI853Integrator]:
        raise ValueError("Only Leapfrog and dop853 integration is supported for"
                         " generating mock streams.")

    if not isinstance(hamiltonian, Hamiltonian) or not hamiltonian.c_enabled:
        raise TypeError("Input potential must be a CPotentialBase subclass.")

    if not isinstance(prog_orbit, Orbit):
        raise TypeError("Progenitor orbit must be an Orbit subclass.")

    if snapshot_filename is not None and Integrator != DOPRI853Integrator:
        raise ValueError("If saving snapshots, must use the DOP853Integrator.")

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

    c_w = np.squeeze(prog_orbit.w(hamiltonian.units)).T # transpose for Cython funcs
    prog_w = np.ascontiguousarray(c_w)
    prog_t = np.ascontiguousarray(prog_orbit.t.decompose(hamiltonian.units).value)
    if hasattr(prog_mass, 'unit'):
        prog_mass = prog_mass.decompose(hamiltonian.units).value

    if Integrator == LeapfrogIntegrator:
        stream_w = _mock_stream_leapfrog(hamiltonian, t=prog_t, prog_w=prog_w,
                                         release_every=release_every,
                                         _k_mean=k_mean, _k_disp=k_disp,
                                         G=hamiltonian.potential.G,
                                         _prog_mass=prog_mass, seed=seed,
                                         **Integrator_kwargs)

    elif Integrator == DOPRI853Integrator:
        if snapshot_filename is not None:
            if os.path.exists(snapshot_filename):
                raise IOError("Mockstream save file '{}' already exists.")

            import h5py

            _mock_stream_animate(snapshot_filename, hamiltonian, t=prog_t, prog_w=prog_w,
                                 release_every=release_every,
                                 _k_mean=k_mean, _k_disp=k_disp,
                                 G=hamiltonian.potential.G,
                                 _prog_mass=prog_mass, seed=seed,
                                 **Integrator_kwargs)

            with h5py.File(str(snapshot_filename), 'a') as h5f:
                h5f['pos'].attrs['unit'] = str(hamiltonian.units['length'])
                h5f['vel'].attrs['unit'] = str(hamiltonian.units['length']/hamiltonian.units['time'])
                h5f['t'].attrs['unit'] = str(hamiltonian.units['time'])

            return None

        else:
            stream_w = _mock_stream_dop853(hamiltonian, t=prog_t, prog_w=prog_w,
                                           release_every=release_every,
                                           _k_mean=k_mean, _k_disp=k_disp,
                                           G=hamiltonian.potential.G,
                                           _prog_mass=prog_mass, seed=seed,
                                           **Integrator_kwargs)

    else:
        raise RuntimeError("Should never get here...")

    return PhaseSpacePosition.from_w(w=stream_w.T, units=hamiltonian.units)

def streakline_stream(hamiltonian, prog_orbit, prog_mass, release_every=1,
                      Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                      snapshot_filename=None, seed=None):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the Streakline method from Kuepper et al. (2012).

    Parameters
    ----------
    hamiltonian : `~gala.potential.Hamiltonian`
        The system Hamiltonian.
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
    snapshot_filename : str (optional)
        Filename to save all incremental snapshots of particle positions and
        velocities. Warning: this can make very large files if you are not
        careful!
    seed : int (optional)
        A random number seed for initializing the particle positions.

    Returns
    -------
    stream : `~gala.dynamics.PhaseSpacePosition`

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

    return mock_stream(hamiltonian=hamiltonian, prog_orbit=prog_orbit, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs,
                       snapshot_filename=snapshot_filename, seed=seed)

def fardal_stream(hamiltonian, prog_orbit, prog_mass, release_every=1,
                  Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                  snapshot_filename=None, seed=None):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the prescription from Fardal et al. (2015).

    Parameters
    ----------
    hamiltonian : `~gala.potential.Hamiltonian`
            The system Hamiltonian.
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
    snapshot_filename : str (optional)
        Filename to save all incremental snapshots of particle positions and
        velocities. Warning: this can make very large files if you are not
        careful!
    seed : int (optional)
        A random number seed for initializing the particle positions.

    Returns
    -------
    stream : `~gala.dynamics.PhaseSpacePosition`

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

    return mock_stream(hamiltonian=hamiltonian, prog_orbit=prog_orbit, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs,
                       snapshot_filename=snapshot_filename, seed=seed)

def dissolved_fardal_stream(hamiltonian, prog_orbit, prog_mass, t_disrupt, release_every=1,
                            Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                            snapshot_filename=None, seed=None):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the prescription from Fardal et al. (2015), but at a specified
    time the progenitor completely dissolves and the radial offset of the
    tidal radius is reduced to 0 at fixed dispersion.

    Parameters
    ----------
    hamiltonian : `~gala.potential.Hamiltonian`
            The system Hamiltonian.
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
    snapshot_filename : str (optional)
        Filename to save all incremental snapshots of particle positions and
        velocities. Warning: this can make very large files if you are not
        careful!
    seed : int (optional)
        A random number seed for initializing the particle positions.

    Returns
    -------
    stream : `~gala.dynamics.PhaseSpacePosition`

    """

    try:
        # the time index closest to when the disruption happens
        t = prog_orbit.t
    except AttributeError:
        raise TypeError("Input progenitor orbit must be an Orbit subclass instance.")

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

    return mock_stream(hamiltonian=hamiltonian, prog_orbit=prog_orbit, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs,
                       snapshot_filename=snapshot_filename, seed=seed)
