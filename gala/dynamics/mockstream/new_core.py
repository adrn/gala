# Standard library
import warnings

# Third-party
import astropy.units as u
import numpy as np

# This package
from .. import combine
from ..nbody import DirectNBody
from ...potential import Hamiltonian, PotentialBase
from ...integrate.timespec import parse_time_specification

__all__ = ['MockStreamGenerator']

class MockStreamGenerator:

    def __init__(self, df, H,
                 progenitor_potential=None,
                 progenitor_mass_loss=None,
                 self_gravity=None):

        # TODO: deal with df
        self.df = df

        # Validate the inpute hamiltonian
        if not isinstance(H, Hamiltonian):
            H = Hamiltonian(H)
        self.H = H

        if progenitor_potential is not None:
            # validate the potential class
            if not isinstance(progenitor_potential, PotentialBase):
                raise TypeError("If specified, the `progenitor_potential` must "
                                "be a gala.potential class instance.")

            if self_gravity is None:
                self_gravity = True

            elif self_gravity is None:
                self_gravity = False
        else:
            if self_gravity is not None:
                warnings.warn("Ignoring self-gravity setting: you must specify "
                              "a progenitor potential if self_gravity=True.")
            self_gravity = False

        self.self_gravity = self_gravity
        self.progenitor_potential = progenitor_potential

        # TODO: only mass-loss currently allowed is uniform mass-loss:
        if (not isinstance(progenitor_mass_loss, u.Quantity) or
                not progenitor_mass_loss.isscalar):
            raise NotImplementedError("The only mass-loss currently supported "
                                      "is uniform mass-loss, specified by "
                                      "passing in an astropy Quantity with "
                                      "unit: <mass unit> / <time unit>")

    def _get_nbody(self, prog_w0, other_w0=None, save_all=True,
                   nbody_kwargs=None):

        # Parse the nbody kwargs: this allows passing in other massive bodies,
        # i.e. perturbers during the mock stream generation
        if nbody_kwargs is None:
            nbody_kwargs = dict()
        nbody_w0 = nbody_kwargs.get('w0', None)
        nbody_pps = nbody_kwargs.get('particle_potentials', None)
        nbody_units = nbody_kwargs.get('units', None)

        if ((nbody_w0 is None and nbody_pps is not None) or
                (nbody_w0 is not None and nbody_pps is None)):
            raise ValueError("If N-body initial conditions are specified, "
                             "you must also specify an equal number of "
                             "N-body particle potentials. The elements "
                             "of these can be `None`, which tells the "
                             "integrator to treat them as test particles.")

        w0 = prog_w0
        if nbody_w0 is not None:
            w0 = combine((w0, nbody_w0))
            pps = [self.progenitor_potential] + list(nbody_pps)
        else:
            pps = [self.progenitor_potential]

        # Test particle initial conditions to include:
        if other_w0 is not None:
            w0 = combine((w0, other_w0))
            pps = pps + [None] * other_w0.shape[0]

        # TODO: possible return slice objects for progenitor, perturbers, particles?
        # OR: at least the number of other massive bodies?
        return DirectNBody(w0=w0, particle_potentials=pps,
                           external_potential=self.H.potential,
                           frame=self.H.frame, units=nbody_units,
                           save_all=save_all)

    def _make_ics(self, prog_orbit, prog_m, ):
        prog_x = np.ascontiguousarray(
            prog_orbit.xyz.decompose(self.H.units).value.T)
        prog_v = np.ascontiguousarray(
            prog_orbit.v_xyz.decompose(self.H.units).value.T)
        prog_t = prog_orbit.t.decompose(self.H.units).value.T

        # HACKS
        prog_m = np.zeros_like(prog_t) + 1e5
        prog_rs = np.zeros_like(prog_t) + 0.01
        n_particles = np.zeros(len(prog_t), dtype='i4') + 1

        stuff = self.df._sample(prog_x, prog_v, prog_t, prog_m, prog_rs, n_particles)

    def run(self, prog_w0, nbody_kwargs=None, **time_spec):
        t = parse_time_specification(self.H.units, **time_spec)

        prog_nbody = self._get_nbody(prog_w0, nbody_kwargs=nbody_kwargs)
        prog_orbit = prog_nbody.integrate_orbit(t=t)

        # If the time stepping passed in is negative, assume this means that all
        # of the initial conditions are at *end time*, and we first need to
        # integrate them backwards before treating them as initial conditions
        # TODO: this does *not* do what is described in this comment ^
        if t[-1] < t[0]:
            prog_orbit = prog_orbit[::-1]

        # TODO: can pre-compute mass vs. time
        prog_mass = np.full(prog_orbit.ntimes, 1e5) * u.Msun # HACK

        # generate initial conditions for stream particles
        w0s, t1s = self._make_ics(prog_orbit, prog_mass, )
        # TODO: flag to ics to ignore particles obviously bound to progenitor?

        # TODO: run integration
        dt = prog_orbit.t[1] - prog_orbit.t[0]
        all_ws = []
        for w0, t1 in zip(w0s, t1s):
            nbody = self._get_nbody(prog_w0, other_w0=w0, save_all=False,
                                    nbody_kwargs=nbody_kwargs)
            if t1 == prog_orbit.t[-1]:
                break

            end_w = nbody.integrate_orbit(dt=dt, t1=t1, t2=prog_orbit.t[-1])
            all_ws.append(end_w[2:]) # HACK

        return combine(all_ws)
