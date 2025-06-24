# cython: boundscheck=False
# cython: debug=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False


import numpy as np

from ...integrate.cyintegrators.leapfrog import leapfrog_integrate_nbody
from ...integrate.cyintegrators.ruth4 import ruth4_integrate_nbody
from ...integrate.timespec import parse_time_specification
from ...potential import Hamiltonian, NullPotential, StaticFrame
from ...units import UnitSystem
from ...util import atleast_2d
from ..core import PhaseSpacePosition
from ..orbit import Orbit
from .nbody import direct_nbody_dop853, nbody_acceleration

__all__ = ["DirectNBody"]


class DirectNBody:
    def __init__(
        self,
        w0,
        particle_potentials,
        external_potential=None,
        frame=None,
        units=None,
        save_all=True,
    ):
        """Perform orbit integration using direct N-body forces between
        particles, optionally in an external background potential.

        TODO: could add another option, like in other contexts, for
        "extra_force" to support, e.g., dynamical friction

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
        frame : :class:`~gala.potential.frame.FrameBase` subclass (optional)
            The reference frame to perform integratiosn in.
        units : `~gala.units.UnitSystem` (optional)
            Set of non-reducable units that specify (at minimum) the
            length, mass, time, and angle units.
        save_all : bool (optional)
            Save the full orbits of each particle. If ``False``, only returns
            the final phase-space positions of each particle.

        """
        if not isinstance(w0, PhaseSpacePosition):
            msg = (
                "Initial conditions `w0` must be a "
                "gala.dynamics.PhaseSpacePosition object, "
                f"not '{w0.__class__.__name__}'"
            )
            raise TypeError(msg)

        if len(w0.shape) > 0 and w0.shape[0] != len(particle_potentials):
            raise ValueError(
                "The number of initial conditions in `w0` must"
                " match the number of particle potentials "
                "passed in with `particle_potentials`."
            )

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
            raise ValueError(
                "Could not determine units from input! You must "
                "either (1) pass in the unit system with `units`,"
                "(2) set the units on one of the "
                "particle_potentials, OR (3) pass in an "
                "`external_potential` with valid units."
            )
        if not isinstance(units, UnitSystem):
            units = UnitSystem(units)

        # Now that we have the unit system, enforce that all potentials are in
        # that system:
        particle_potentials_ = []
        for pp in particle_potentials:
            pp = NullPotential(units=units) if pp is None else pp.replace_units(units)
            particle_potentials_.append(pp)

        if external_potential is None:
            external_potential = NullPotential(units=units)
        else:
            external_potential = external_potential.replace_units(units)

        if frame is None:
            frame = StaticFrame(units)

        self.units = units
        self.external_potential = external_potential
        self.frame = frame
        self.particle_potentials = particle_potentials_
        self.save_all = save_all

        self.H = Hamiltonian(self.external_potential, frame=self.frame)
        if not self.H.c_enabled:
            raise ValueError(
                "Input potential must be C-enabled: one or more "
                "components in the input external potential are "
                "Python-only."
            )

        self.w0 = w0

    @property
    def w0(self):
        return self._w0

    @w0.setter
    def w0(self, value):
        self._w0 = value
        self._cache_w0()

    def _cache_w0(self):
        # cache the position and velocity / prepare the initial conditions
        self._pos = atleast_2d(self.w0.xyz.decompose(self.units).value, insert_axis=1)
        self._vel = atleast_2d(self.w0.v_xyz.decompose(self.units).value, insert_axis=1)
        self._c_w0 = np.ascontiguousarray(np.vstack((self._pos, self._vel)).T)

    def __repr__(self):
        if self.w0.shape:
            return f"<{self.__class__.__name__} bodies={self.w0.shape[0]}>"
        return f"<{self.__class__.__name__} bodies=1>"

    def _nbody_acceleration(self, t=0.0):
        """
        Compute the N-body acceleration at the location of each body
        """
        nbody_acc = nbody_acceleration(self._c_w0, t, self.particle_potentials)
        return nbody_acc.T

    def acceleration(self, t=0.0):
        """
        Compute the acceleration at the location of each N body, including the
        external potential.
        """
        nbody_acc = self._nbody_acceleration(t=t) * self.units["acceleration"]
        ext_acc = self.external_potential.acceleration(self.w0, t=t)
        return nbody_acc + ext_acc

    def integrate_orbit(self, Integrator=None, Integrator_kwargs=None, **time_spec):
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
        from gala.integrate import (
            DOPRI853Integrator,
            LeapfrogIntegrator,
            Ruth4Integrator,
        )

        if Integrator_kwargs is None:
            Integrator_kwargs = {}
        if Integrator is None:
            Integrator = DOPRI853Integrator

        # Prepare the time-stepping array
        t = parse_time_specification(self.units, **time_spec)

        # Reorganize orbits so that massive bodies are first:
        front_idx = []
        front_pp = []
        end_idx = []
        end_pp = []
        for i, pp in enumerate(self.particle_potentials):
            if not isinstance(pp, NullPotential):
                front_idx.append(i)
                front_pp.append(pp)
            else:
                end_idx.append(i)
                end_pp.append(pp)
        idx = np.array(front_idx + end_idx)
        pps = front_pp + end_pp

        reorg_w0 = np.ascontiguousarray(self._c_w0[idx])

        if Integrator == LeapfrogIntegrator:
            _, ws = leapfrog_integrate_nbody(
                self.H,
                reorg_w0,
                t,
                pps,
                save_all=int(self.save_all),
                **Integrator_kwargs,
            )
        elif Integrator == Ruth4Integrator:
            _, ws = ruth4_integrate_nbody(
                self.H,
                reorg_w0,
                t,
                pps,
                save_all=int(self.save_all),
                **Integrator_kwargs,
            )
        elif Integrator == DOPRI853Integrator:
            ws = direct_nbody_dop853(
                reorg_w0, t, self.H, pps, save_all=self.save_all, **Integrator_kwargs
            )
        else:
            raise NotImplementedError(
                f"N-body integration is currently not supported with the {Integrator} "
                "integrator class"
            )

        if self.save_all:
            pos = np.rollaxis(np.array(ws[..., :3]), axis=2)  # should this be axis=-1?
            vel = np.rollaxis(np.array(ws[..., 3:]), axis=2)

            orbits = Orbit(
                pos=pos * self.units["length"],
                vel=vel * self.units["length"] / self.units["time"],
                t=t * self.units["time"],
                hamiltonian=self.H,
            )

        else:
            pos = np.array(ws[..., :3]).T
            vel = np.array(ws[..., 3:]).T

            orbits = PhaseSpacePosition(
                pos=pos * self.units["length"],
                vel=vel * self.units["length"] / self.units["time"],
                frame=self.frame,
            )

        # Reorder orbits to original order:
        undo_idx = np.argsort(idx)

        return orbits[..., undo_idx]
