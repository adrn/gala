# Standard library
import warnings
import os

# Third-party
import numpy as np

# Project
from .. import PhaseSpacePosition, Orbit
from ...potential import Hamiltonian, CPotentialBase
from ...integrate import DOPRI853Integrator, LeapfrogIntegrator
from ._mockstream import (_mock_stream_dop853, _mock_stream_leapfrog,
                          _mock_stream_animate)

__all__ = ['mock_stream', 'streakline_stream', 'fardal_stream',
           'dissolved_fardal_stream']


def mock_stream(hamiltonian, prog_orbit, prog_mass, k_mean, k_disp,
                release_every=1, Integrator=DOPRI853Integrator,
                Integrator_kwargs=dict(),
                snapshot_filename=None, output_every=1, seed=None):
    """DEPRECATED!"""
    raise NotImplementedError(
        "This function has been deprecated by the new mock stream generation "
        "methodology. See "
        "http://gala.adrian.pw/en/latest/dynamics/mockstreams.html for "
        "information about the new functionality.")


def streakline_stream(hamiltonian, prog_orbit, prog_mass, release_every=1,
                      Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                      snapshot_filename=None, output_every=1, seed=None):
    """
    This function has been deprecated! TODO: link to transition guide...

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
    output_every : int (optional)
        If outputing snapshots (i.e., if snapshot_filename is specified), this
        controls how often to output a snapshot.
    seed : int (optional)
        A random number seed for initializing the particle positions.

    Returns
    -------
    stream : `~gala.dynamics.PhaseSpacePosition`

    """
    pass


def fardal_stream(hamiltonian, prog_orbit, prog_mass, release_every=1,
                  Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                  snapshot_filename=None, seed=None, output_every=1):
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
    output_every : int (optional)
        If outputing snapshots (i.e., if snapshot_filename is specified), this
        controls how often to output a snapshot.
    seed : int (optional)
        A random number seed for initializing the particle positions.

    Returns
    -------
    stream : `~gala.dynamics.PhaseSpacePosition`

    """
    pass


def dissolved_fardal_stream(hamiltonian, prog_orbit, prog_mass, t_disrupt, release_every=1,
                            Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                            snapshot_filename=None, output_every=1, seed=None):
    """DEPRECATED!"""
    raise NotImplementedError(
        "This function has been deprecated by the new mock stream generation "
        "methodology. See "
        "http://gala.adrian.pw/en/latest/dynamics/mockstreams.html for "
        "information about the new functionality.")
