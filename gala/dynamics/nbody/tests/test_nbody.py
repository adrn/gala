# Third-party
import astropy.units as u
import numpy as np
import pytest

# Custom
from ....potential import (NullPotential, NFWPotential,
                           HernquistPotential, KuzminPotential,
                           ConstantRotatingFrame, StaticFrame)
from ....dynamics import PhaseSpacePosition, combine
from ....units import UnitSystem, galactic

# Project
from ..core import DirectNBody

class TestDirectNBody:

    def setup(self):
        self.usys = UnitSystem(u.pc, u.Unit(1e-5*u.Myr),
                               u.Unit(1e6*u.Msun), u.radian)
        pot_particle2 = HernquistPotential(m=1e6*u.Msun, c=0.1*u.pc,
                                           units=self.usys)
        vcirc = pot_particle2.circular_velocity([1, 0, 0.]*u.pc).to(u.km/u.s)

        self.particle_potentials = [NullPotential(self.usys), pot_particle2]

        w0_2 = PhaseSpacePosition(pos=[10, 0, 0] * u.kpc,
                                  vel=[0, 83, 0] * u.km/u.s)
        w0_1 = PhaseSpacePosition(pos=w0_2.xyz + [1, 0, 0] * u.pc,
                                  vel=w0_2.v_xyz + [0, 1., 0] * vcirc)
        self.w0 = combine((w0_1, w0_2))

        self.ext_pot = NFWPotential(m=1e11, r_s=10, units=galactic)

    def test_directnbody_init(self):
        # another unit system for testing
        usys2 = UnitSystem(u.pc, u.Unit(1e-3*u.Myr),
                           u.Unit(1e6*u.Msun), u.radian)

        particle_potentials_None = [None] + self.particle_potentials[1:]

        # Different VALID ways to initialize
        nbody = DirectNBody(self.w0,
                            particle_potentials=self.particle_potentials)
        nbody = DirectNBody(self.w0,
                            particle_potentials=particle_potentials_None)
        nbody = DirectNBody(self.w0,
                            particle_potentials=self.particle_potentials,
                            external_potential=self.ext_pot)
        nbody = DirectNBody(self.w0,
                            particle_potentials=self.particle_potentials,
                            external_potential=self.ext_pot, units=usys2)
        nbody = DirectNBody(self.w0, particle_potentials=[None, None],
                            units=usys2)
        nbody = DirectNBody(self.w0, particle_potentials=[None, None],
                            external_potential=self.ext_pot)

        # Different INVALID ways to initialize
        with pytest.raises(TypeError):
            DirectNBody("sdf", particle_potentials=self.particle_potentials)

        with pytest.raises(ValueError):
            DirectNBody(self.w0,
                        particle_potentials=self.particle_potentials[:1])

        # MAX_NBODY1 = 65536+1
        # w0_max = combine([self.w0[0]]*MAX_NBODY1)
        # with pytest.raises(NotImplementedError):
        #     DirectNBody(w0_max, particle_potentials=[None]*MAX_NBODY1)

        with pytest.raises(ValueError):
            DirectNBody(self.w0, particle_potentials=[None, None])

        py_ext_pot = KuzminPotential(m=1e10*u.Msun, a=0.5*u.kpc, units=galactic)
        with pytest.raises(ValueError):
            DirectNBody(self.w0, particle_potentials=self.particle_potentials,
                        external_potential=py_ext_pot)

    def test_directnbody_integrate(self):
        # TODO: this is really a unit test, but we should have some functional tests
        # that check that the orbit integration is making sense!

        # First, compare with/without mass with no external potential:
        nbody1 = DirectNBody(self.w0,
                             particle_potentials=[None, None],
                             units=self.usys)
        nbody2 = DirectNBody(self.w0,
                             particle_potentials=self.particle_potentials,
                             units=self.usys)

        orbits1 = nbody1.integrate_orbit(dt=1*self.usys['time'],
                                         t1=0, t2=1*u.Myr)
        orbits2 = nbody2.integrate_orbit(dt=1*self.usys['time'],
                                         t1=0, t2=1*u.Myr)

        dx0 = orbits1[:, 0].xyz - orbits2[:, 0].xyz
        dx1 = orbits1[:, 1].xyz - orbits2[:, 1].xyz
        assert u.allclose(np.abs(dx1), 0*u.pc, atol=1e-13*u.pc)
        assert np.abs(dx0).max() > 50*u.pc

        # Now compare with/without mass with external potential:
        nbody1 = DirectNBody(self.w0,
                             particle_potentials=[None, None],
                             units=self.usys,
                             external_potential=self.ext_pot)
        nbody2 = DirectNBody(self.w0,
                             particle_potentials=self.particle_potentials,
                             units=self.usys,
                             external_potential=self.ext_pot)

        orbits1 = nbody1.integrate_orbit(dt=1*self.usys['time'],
                                         t1=0, t2=1*u.Myr)
        orbits2 = nbody2.integrate_orbit(dt=1*self.usys['time'],
                                         t1=0, t2=1*u.Myr)

        dx0 = orbits1[:, 0].xyz - orbits2[:, 0].xyz
        dx1 = orbits1[:, 1].xyz - orbits2[:, 1].xyz
        assert u.allclose(np.abs(dx1), 0*u.pc, atol=1e-13*u.pc)
        assert np.abs(dx0).max() > 50*u.pc

    def test_directnbody_integrate_dontsaveall(self):
        # If we set save_all = False, only return the final positions:
        nbody1 = DirectNBody(self.w0,
                             particle_potentials=self.particle_potentials,
                             units=self.usys,
                             external_potential=self.ext_pot,
                             save_all=False)
        nbody2 = DirectNBody(self.w0,
                             particle_potentials=self.particle_potentials,
                             units=self.usys,
                             external_potential=self.ext_pot,
                             save_all=True)

        w1 = nbody1.integrate_orbit(dt=1*self.usys['time'],
                                    t1=0, t2=1*u.Myr)
        orbits = nbody2.integrate_orbit(dt=1*self.usys['time'],
                                        t1=0, t2=1*u.Myr)
        w2 = orbits[-1]
        assert u.allclose(w1.xyz, w2.xyz)
        assert u.allclose(w1.v_xyz, w2.v_xyz)

    def test_directnbody_integrate_rotframe(self):
        # Now compare with/without mass with external potential:
        frame = ConstantRotatingFrame(Omega=[0,0,1]*self.w0[0].v_y/self.w0[0].x,
                                      units=self.usys)
        nbody = DirectNBody(self.w0,
                            particle_potentials=self.particle_potentials,
                            units=self.usys,
                            external_potential=self.ext_pot,
                            frame=frame)
        nbody2 = DirectNBody(self.w0,
                             particle_potentials=self.particle_potentials,
                             units=self.usys,
                             external_potential=self.ext_pot)

        orbits = nbody.integrate_orbit(dt=1*self.usys['time'],
                                       t1=0, t2=1*u.Myr)
        orbits_static = orbits.to_frame(StaticFrame(self.usys))

        orbits2 = nbody2.integrate_orbit(dt=1*self.usys['time'],
                                         t1=0, t2=1*u.Myr)

        assert u.allclose(orbits_static.xyz, orbits_static.xyz)
        assert u.allclose(orbits2.v_xyz, orbits2.v_xyz)
