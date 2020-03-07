# Standard library
import warnings

# Third-party
import astropy.units as u
import numpy as np

# Project
from ...integrate import DOPRI853Integrator
from ...io import quantity_to_hdf5, quantity_from_hdf5
from .. import PhaseSpacePosition

__all__ = ['MockStream',
           'mock_stream', 'streakline_stream', # DEPRECATED: TODO remove
           'fardal_stream', 'dissolved_fardal_stream']

_transition_guide_url = "http://gala.adrian.pw/en/latest/dynamics/mockstreams.html"


class MockStream(PhaseSpacePosition):

    @u.quantity_input(release_time=u.Myr)
    def __init__(self, pos, vel=None, frame=None,
                 release_time=None, lead_trail=None):

        super().__init__(pos=pos, vel=vel, frame=frame)

        if release_time is not None:
            release_time = u.Quantity(release_time)
            if len(release_time) != self.pos.shape[0]:
                raise ValueError('shape mismatch: input release time array '
                                 'must have the same shape as the input '
                                 'phase-space data, minus the component '
                                 'dimension. expected {}, got {}'
                                 .format(self.pos.shape[0],
                                         len(release_time)))

        self.release_time = release_time

        if lead_trail is not None:
            lead_trail = np.array(lead_trail)
            if len(lead_trail) != self.pos.shape[0]:
                raise ValueError('shape mismatch: input leading/trailing array '
                                 'must have the same shape as the input '
                                 'phase-space data, minus the component '
                                 'dimension. expected {}, got {}'
                                 .format(self.pos.shape[0],
                                         len(lead_trail)))

        self.lead_trail = lead_trail

    # ------------------------------------------------------------------------
    # Input / output
    #
    def to_hdf5(self, f):
        """
        Serialize this object to an HDF5 file.

        Requires ``h5py``.

        Parameters
        ----------
        f : str, :class:`h5py.File`
            Either the filename or an open HDF5 file.
        """

        f = super().to_hdf5(f)

        # if self.potential is not None:
        #     import yaml
        #     from ..potential.potential.io import to_dict
        #     f['potential'] = yaml.dump(to_dict(self.potential)).encode('utf-8')

        if self.release_time:
            quantity_to_hdf5(f, 'release_time', self.release_time)

        if self.lead_trail is not None:
            f['lead_trail'] = self.lead_trail.astype('S1') # TODO HACK
        return f

    @classmethod
    def from_hdf5(cls, f):
        """
        Load an object from an HDF5 file.

        Requires ``h5py``.

        Parameters
        ----------
        f : str, :class:`h5py.File`
            Either the filename or an open HDF5 file.
        """
        # TODO: this is duplicated code from PhaseSpacePosition
        if isinstance(f, str):
            import h5py
            f = h5py.File(f, mode='r')

        obj = PhaseSpacePosition.from_hdf5(f)

        if 'release_time' in f:
            t = quantity_from_hdf5(f['release_time'])
        else:
            t = None

        if 'lead_trail' in f:
            lt = f['lead_trail'][:]
        else:
            lt = None

        return cls(pos=obj.pos, vel=obj.vel,
                   release_time=t, lead_trail=lt,
                   frame=obj.frame)


# ---------------------------------------------------------------------------
# DEPRECATED / OLD STUFF BELOW
# TODO: remove this

def mock_stream(hamiltonian, prog_orbit, prog_mass, k_mean, k_disp,
                release_every=1, Integrator=DOPRI853Integrator,
                Integrator_kwargs=dict(),
                snapshot_filename=None, output_every=1, seed=None):
    """DEPRECATED!"""
    raise NotImplementedError(
        "This function has been deprecated by the new mock stream generation "
        "methodology. See {} for information about the new functionality."
        .format(_transition_guide_url))


def streakline_stream(hamiltonian, prog_orbit, prog_mass, release_every=1,
                      Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                      snapshot_filename=None, output_every=1, seed=None):
    """This function has been deprecated!

    See {} for more information.

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
    from .df import StreaklineStreamDF
    from .mockstream_generator import MockStreamGenerator

    warnings.warn("This function is deprecated - use the new mock stream "
                  "generation functionality. See {} for more information."
                  .format(_transition_guide_url), DeprecationWarning)

    if Integrator is not DOPRI853Integrator:
        raise ValueError("Integrator must be DOPRI853Integrator.")

    rnd = np.random.RandomState(seed)
    df = StreaklineStreamDF(random_state=rnd)

    gen = MockStreamGenerator(df=df, hamiltonian=hamiltonian)
    stream, _ = gen.run(prog_orbit[0], prog_mass=prog_mass,
                        release_every=release_every, t=prog_orbit.t)

    return stream


def fardal_stream(hamiltonian, prog_orbit, prog_mass, release_every=1,
                  Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                  snapshot_filename=None, seed=None, output_every=1):
    """This function has been deprecated!

    See {} for more information.

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
    from .df import FardalStreamDF
    from .mockstream_generator import MockStreamGenerator
    warnings.warn("This function is deprecated - use the new mock stream "
                  "generation functionality. See {} for more information."
                  .format(_transition_guide_url), DeprecationWarning)

    if Integrator is not DOPRI853Integrator:
        raise ValueError("Integrator must be DOPRI853Integrator.")

    rnd = np.random.RandomState(seed)
    df = FardalStreamDF(random_state=rnd)

    gen = MockStreamGenerator(df=df, hamiltonian=hamiltonian)
    stream, _ = gen.run(prog_orbit[0], prog_mass=prog_mass,
                        release_every=release_every, t=prog_orbit.t)

    return stream


def dissolved_fardal_stream(hamiltonian, prog_orbit, prog_mass, t_disrupt, release_every=1,
                            Integrator=DOPRI853Integrator, Integrator_kwargs=dict(),
                            snapshot_filename=None, output_every=1, seed=None):
    """DEPRECATED!"""
    raise NotImplementedError(
        "This function has been deprecated by the new mock stream generation "
        "methodology. See {} for information about the new functionality."
        .format(_transition_guide_url))


streakline_stream.__doc__ = streakline_stream.__doc__.format(_transition_guide_url)
fardal_stream.__doc__ = fardal_stream.__doc__.format(_transition_guide_url)
