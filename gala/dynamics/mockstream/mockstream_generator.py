# Third-party
import numpy as np

# This package
from .. import combine, PhaseSpacePosition
from ..nbody import DirectNBody
from ...potential import Hamiltonian, PotentialBase
from ...integrate.timespec import parse_time_specification
from .df import BaseStreamDF
from .mockstream import mockstream_dop853

__all__ = ['MockStreamGenerator']

class MockStreamGenerator:

    def __init__(self, df, hamiltonian, progenitor_potential=None):
        """Generate a mock stellar stream in the specified external potential.

        By default, you must pass in a specification of the stream distribution
        function (``df``), and the external gravitational potential and
        reference frame (via a `~gala.potential.Hamiltonian` object passed in
        through the ``hamiltonian`` argument).

        Also by default, the stream generation does not include the self-gravity
        of the progenitor system: star particles are generated using the ``df``
        object, and released into the external potential specified by the
        ``hamiltonian``. If you would like the star particles to feel the
        gravitational field of the progenitor system, you may pass in a
        potential object to represent the progenitor via the
        ``progenitor_potential`` argument. This can be any valid gala potential
        instance.

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

    def run(self, prog_w0, prog_mass, nbody=None,
            release_every=1, n_particles=1,
            **time_spec):
        """Run the mock stream generator with the specified progenitor initial
        conditions.

        This method generates the mock stellar stream for the specified
        progenitor system properties. The progenitor orbit is specified by
        passing in the initial or final conditions ``prog_w0`` and by specifying
        time-stepping information via the ``**time_spec`` keyword arguments. If
        the time-stepping specification proceeds forward in time, ``prog_w0`` is
        interpreted as initial conditions and the mock stream is generated
        forwards from this position. If the time-stepping proceeds backwards in
        time, the progenitor orbit is first numerically integrated backwards
        given the time-stepping information, then the stream is generated
        forward from the past such that ``prog_w0`` becomes the final position
        of the progenitor.

        Note that the stream generation also supports including other massive
        perturbers that can gravitationally influence the stream stars. These
        other massive bodies must be passed in as a `~gala.dynamics.DirectNBody`
        instance through the ``nbody`` argument. The phase-space coordinates of
        the bodies, ``nbody.w0``, are interpreted as initial or final conditions
        with the same logic as above.

        Parameters
        ----------
        prog_w0 : `~gala.dynamics.PhaseSpacePosition`
            The initial or final phase-space position of the progenitor system
            (see note above).
        prog_mass : `~astropy.units.Quantity` [mass]
            The mass of the progenitor system, passed in to the stream
            distribution function (df) ``.sample()`` method. This quantity sets
            the scale mass of the particle release df, but not the mass of the
            progenitor potential used to compute the self-gravity on the stream
            particles.
        nbody : `~gala.dynamics.DirectNBody` (optional)
            This allows specifying other massive perturbers (N-bodies) that can
            gravitationally influence the stream star orbits.
        release_every : int (optional)
            Controls how often to release stream particles from each tail.
            Default: 1, meaning release particles at each timestep.
        n_particles : int, array_like (optional)
            If an integer, this controls the number of particles to release in
            each tail at each release timestep. Alternatively, you can pass in
            an array with the same shape as the number of timesteps to release
            bursts of particles at certain times (e.g., pericenter).
        **time_spec
            Specification of how long to integrate. Most commonly, this is a
            timestep ``dt`` and number of steps ``n_steps``, or a timestep
            ``dt``, initial time ``t1``, and final time ``t2``. You may also
            pass in a time array with ``t``. See documentation for
            `~gala.integrate.parse_time_specification` for more information.

        Returns
        -------
        stream_w : `~gala.dynamics.PhaseSpacePosition`
        nbody_w : `~gala.dynamics.PhaseSpacePosition`

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

        prog_orbit = nbody_orbits[:, 0] # Note: Progenitor must be idx 0!

        # Generate initial conditions from the DF
        stream_w0 = self.df.sample(prog_orbit, prog_mass,
                                   hamiltonian=self.hamiltonian,
                                   release_every=release_every,
                                   n_particles=n_particles)
        w0 = np.vstack((stream_w0.xyz.decompose(units).value,
                        stream_w0.v_xyz.decompose(units).value)).T
        w0 = np.ascontiguousarray(w0)

        unq_t1s, nstream = np.unique(stream_w0.t.decompose(units),
                                     return_counts=True)

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
