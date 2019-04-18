# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

# Third-party
import numpy as np

from ...potential import Hamiltonian, NullPotential
from ...units import UnitSystem
from ...integrate.timespec import parse_time_specification
from .. import Orbit, PhaseSpacePosition

from ._nbody import _direct_nbody_dop853

__all__ = ['DirectNBody']


class DirectNBody:

    def __init__(self, w0, particle_potentials, external_potential=None,
                 units=None, save_all=True):
        """Perform orbit integration using direct N-body forces between
        particles, optionally in an external background potential.

        TODO: could add another option, like in other contexts, for "extra_force"
        to support, e.g., dynamical friction

        Parameters
        ----------
        w0 : `~gala.dynamics.PhaseSpacePosition`
            The particle initial conditions.
        partcle_potentials : list
            List of potential objects to add mass or mass distributions to the
            particles. Use ``None`` to treat particles as test particles.
        external_potential : `~gala.potential.PotentialBase` subclass instance (optional)
            The background or external potential to integrate the particle
            orbits in.
        units : `~gala.units.UnitSystem` (optional)
            Set of non-reducable units that specify (at minimum) the
            length, mass, time, and angle units.
        save_all : bool (optional)
            Save the full orbits of each particle. If ``False``, only returns
            the final phase-space positions of each particle.

        """
        if not isinstance(w0, PhaseSpacePosition):
            raise TypeError("Initial conditions `w0` must be a "
                            "gala.dynamics.PhaseSpacePosition object, "
                            "not '{}'".format(w0.__class__.__name__))

        nbodies = w0.shape[0]
        if not nbodies == len(particle_potentials):
            raise ValueError("The number of initial conditions in `w0` must "
                             "match the number of particle potentials passed "
                             "in with `particle_potentials`.")

        # TODO: this is a MAJOR HACK
        if nbodies > 65536: # see MAX_NBODY in _nbody.pyx
            raise NotImplementedError("We currently only support direct N-body "
                                      "integration for <= 65536 particles.")

        # First, figure out how to get units - first place to check is the arg
        if units is None:
            # Next, check the particle potentials
            for pp in particle_potentials:
                if pp is not None:
                    units = pp.units
                    break

        # If units is still none, and external_potential is defined, use that:
        if units is None and external_potential is not None:
            units = external_potential.units

        # Now, if units are still None, raise an error!
        if units is None:
            raise ValueError("Could not determine units from input! You must "
                             "either (1) pass in the unit system with `units`,"
                             "(2) set the units on one of the "
                             "particle_potentials, OR (3) pass in an "
                             "`external_potential` with valid units.")
        units = UnitSystem(units)

        # Now that we have the unit system, enforce that all potentials are in
        # that system:
        _particle_potentials = []
        for pp in particle_potentials:
            if pp is None:
                pp = NullPotential(units)
            else:
                pp = pp.replace_units(units, copy=True)
            _particle_potentials.append(pp)

        if external_potential is None:
            external_potential = NullPotential(units)
        else:
            external_potential = external_potential.replace_units(units,
                                                                  copy=True)

        self.w0 = w0
        self.units = units
        self.external_potential = external_potential
        self.particle_potentials = _particle_potentials
        self.save_all = save_all

        # This currently only supports non-rotating frames
        self._ext_ham = Hamiltonian(self.external_potential)
        if not self._ext_ham.c_enabled:
            raise ValueError("Input potential must be C-enabled: one or more "
                             "components in the input external potential are "
                             "Python-only.")

    def __repr__(self):
        return "<{} bodies={}>".format(self.__class__.__name__,
                                       self.w0.shape[0])

    def integrate_orbit(self, **time_spec):
        """
        Integrate the initial conditions in the combined external potential
        plus N-body forces.

        This integration uses the `~gala.integrate.DOPRI853Integrator`.

        Parameters
        ----------
        **time_spec
            Specification of how long to integrate. See documentation
            for `~gala.integrate.parse_time_specification`.

        Returns
        -------
        orbit : `~gala.dynamics.Orbit`
            The orbits of the particles.

        """

        # Prepare the initial conditions
        pos = self.w0.xyz.decompose(self.units).value
        vel = self.w0.v_xyz.decompose(self.units).value
        w0 = np.ascontiguousarray(np.vstack((pos, vel)).T)

        # Prepare the time-stepping array
        t = parse_time_specification(self.units, **time_spec)

        ws = _direct_nbody_dop853(w0, t, self._ext_ham,
                                  self.particle_potentials,
                                  save_all=self.save_all)

        if self.save_all:
            pos = np.rollaxis(np.array(ws[..., :3]), axis=2)
            vel = np.rollaxis(np.array(ws[..., 3:]), axis=2)

            orbits = Orbit(
                pos=pos * self.units['length'],
                vel=vel * self.units['length'] / self.units['time'],
                t=t * self.units['time'])

        else:
            pos = np.array(ws[..., :3]).T
            vel = np.array(ws[..., 3:]).T

            orbits = PhaseSpacePosition(
                pos=pos * self.units['length'],
                vel=vel * self.units['length'] / self.units['time'])

        return orbits
