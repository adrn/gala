# Standard library
import warnings

# Third-party
import astropy.units as u
import numpy as np

# This package
from .. import combine, PhaseSpacePosition
from ..nbody import DirectNBody
from ...potential import Hamiltonian, PotentialBase
from ...integrate.timespec import parse_time_specification
from .df import BaseStreamDF
from .new_mockstream import mockstream_dop853

__all__ = ['MockStreamGenerator']

class MockStreamGenerator:

    def __init__(self, df, hamiltonian,
                 progenitor_potential=None,
                 release_every=1, n_particles=1):
        """Generate a mock stellar stream in the specified Hamiltonian.

        Parameters
        ----------
        df : `~gala.dynamics.BaseStreamDF` subclass instance
            The stream distribution function (DF) object that specifies how to
            generate stream star particle initial conditions.
        hamiltonian : `~gala.potential.Hamiltonian`
            The external potential and reference frame to numerically integrate
            orbits in.
        progenitor_potential : `~gala.potential.PotentialBase` (optional)
            If specified, the self-gravity of the progenitor system is included
            in the force calculation and orbit integration. If not specified,
            self-gravity is not accounted for. Default: ``None``
        release_every : int (optional)
            Controls how often to release stream particles from each tail.
            Default: 1, meaning release particles at each timestep.
        n_particles : int, array_like (optional)
            If an integer, this controls the number of particles to release in
            each tail at each release timestep. Alternatively, you can pass in
            an array with the same shape as the number of timesteps to release
            bursts of particles at certain times (e.g., pericenter).
        """

        if not isinstance(df, BaseStreamDF):
            raise TypeError('The input distribution function (DF) instance '
                            'must be an instance of a subclass of '
                            'BaseStreamDF, not {}.'.format(type(df)))
        self.df = df

        # Validate the inpute hamiltonian
        self.hamiltonian = Hamiltonian(hamiltonian)

        if progenitor_potential is not None:
            # validate the potential class
            if not isinstance(progenitor_potential, PotentialBase):
                raise TypeError("If specified, the progenitor_potential must "
                                "be a gala.potential class instance.")

            self.self_gravity = True

        else:
            self.self_gravity = False

        self.progenitor_potential = progenitor_potential
        self.release_every = release_every
        self.n_particles = n_particles

    def _get_nbody(self, prog_w0, nbody):
        """Internal function that adds the progenitor to the list of nbody
        objects to integrate along with the test particles in the stream.
        """

        kwargs = dict()
        if nbody is not None:
            if nbody.external_potential != self.hamiltonian.potential:
                raise ValueError('The external potential of the input nbody '
                                 'instance must match the potential of the mock '
                                 'stream input hamiltonian! {} vs. {}'
                                 .format(nbody.external_potential,
                                         self.hamiltonian.potential))

            if nbody.frame != self.hamiltonian.frame:
                raise ValueError('The reference frame of the input nbody '
                                 'instance must match the frame of the mock '
                                 'stream input hamiltonian! {} vs. {}'
                                 .format(nbody.frame, self.hamiltonian.frame))

            kwargs['w0'] = combine((prog_w0, nbody.w0))
            kwargs['particle_potentials'] = ([self.progenitor_potential] +
                                             nbody.particle_potentials)
            kwargs['external_potential'] = self.hamiltonian.potential
            kwargs['frame'] = self.hamiltonian.frame
            kwargs['units'] = self.hamiltonian.units
            kwargs['save_all'] = nbody.save_all

        else:
            kwargs['w0'] = prog_w0
            kwargs['particle_potentials'] = [self.progenitor_potential]
            kwargs['external_potential'] = self.hamiltonian.potential
            kwargs['frame'] = self.hamiltonian.frame
            kwargs['units'] = self.hamiltonian.units

        return DirectNBody(**kwargs)

    def run(self, prog_w0, prog_mass, nbody=None, **time_spec):
        """Run the mock stream generator with the specified progenitor initial
        conditions.

        TODO: describe nbody stuff...

        Parameters
        ----------
        prog_w0 : `~gala.dynamics.PhaseSpacePosition`
            The initial or final phase-space position of the progenitor system.
            If the time-stepping specification (see below) proceeds forward in
            time, ``prog_w0`` is interpreted as initial conditions and the mock
            stream is generated forwards. If the time-stepping proceeds
            backwards in time, the progenitor orbit is first numerically
            integrated backwards given the time-stepping information, then the
            stream is generated forward from the past such that ``prog_w0``
            becomes the final position of the progenitor.
        prog_mass : `~astropy.units.Quantity` [mass]
            TODO:
        nbody : `~gala.dynamics.DirectNBody` (optional)
            This allows specifying other massive perturbers (N-bodies) that can
            gravitationally influence the stream star orbits.
        **time_spec
            TODO:

        Returns
        -------
        TODO

        """
        units = self.hamiltonian.units
        t = parse_time_specification(units, **time_spec)

        prog_nbody = self._get_nbody(prog_w0, nbody)
        nbody_orbits = prog_nbody.integrate_orbit(t=t)

        # If the time stepping passed in is negative, assume this means that all
        # of the initial conditions are at *end time*, and we first need to
        # integrate them backwards before treating them as initial conditions
        if t[-1] < t[0]:
            nbody_orbits = nbody_orbits[::-1]

            # TODO: this could be cleaed up...
            nbody0 = DirectNBody(
                nbody_orbits[0], prog_nbody.particle_potentials,
                external_potential=prog_nbody.external_potential,
                frame=prog_nbody.frame, units=units)

        else:
            nbody0 = prog_nbody

        prog_orbit = nbody_orbits[:, 0] # Note: assumption! Progenitor is idx 0

        # Generate initial conditions from the DF
        x0, v0, t1 = self.df.sample(prog_orbit, prog_mass,
                                    release_every=self.release_every,
                                    n_particles=self.n_particles)
        w0 = np.hstack((x0.value, v0.value))

        unq_t1s, nstream = np.unique(t1, return_counts=True)

        # Only both iterating over timesteps if we're releasing particles then:
        time = prog_orbit.t.decompose(units).value.copy()
        time = time[np.isin(time, unq_t1s)]

        raw_nbody, raw_stream = mockstream_dop853(nbody0, time, w0, unq_t1s,
                                                  nstream.astype('i4'))

        x_unit = units['length']
        v_unit = units['length'] / units['time']
        stream_w = PhaseSpacePosition(pos=raw_stream[:, :3].T * x_unit,
                                      vel=raw_stream[:, 3:].T * v_unit)
        nbody_w = PhaseSpacePosition(pos=raw_nbody[:, :3].T * x_unit,
                                     vel=raw_nbody[:, 3:].T * v_unit)

        return stream_w, nbody_w
