import astropy.units as u
import numpy as np
import pytest

from gala.dynamics import PhaseSpacePosition, combine
from gala.integrate import (
    DOPRI853Integrator,
    LeapfrogIntegrator,
    Ruth4Integrator,
)

# Custom
from gala.potential import (
    ConstantRotatingFrame,
    HernquistPotential,
    NFWPotential,
    NullPotential,
    StaticFrame,
)
from gala.units import UnitSystem, galactic

from ..core import DirectNBody


class TestDirectNBody:
    def setup_method(self):
        self.usys = UnitSystem(
            u.pc, u.Unit(1e-5 * u.Myr), u.Unit(1e6 * u.Msun), u.radian
        )
        pot_particle2 = HernquistPotential(
            m=1e6 * u.Msun, c=0.1 * u.pc, units=self.usys
        )
        vcirc = pot_particle2.circular_velocity([1, 0, 0.0] * u.pc).to(u.km / u.s)

        self.particle_potentials = [NullPotential(units=self.usys), pot_particle2]

        w0_2 = PhaseSpacePosition(pos=[10, 0, 0] * u.kpc, vel=[0, 83, 0] * u.km / u.s)
        w0_1 = PhaseSpacePosition(
            pos=w0_2.xyz + [1, 0, 0] * u.pc, vel=w0_2.v_xyz + [0, 1.0, 0] * vcirc
        )
        self.w0 = combine((w0_1, w0_2))

        self.ext_pot = NFWPotential(m=1e11, r_s=10, units=galactic)

    def test_directnbody_init(self):
        # another unit system for testing
        usys2 = UnitSystem(u.pc, u.Unit(1e-3 * u.Myr), u.Unit(1e6 * u.Msun), u.radian)

        particle_potentials_None = [None, *self.particle_potentials[1:]]

        # Different VALID ways to initialize
        nbody = DirectNBody(self.w0, particle_potentials=self.particle_potentials)
        nbody = DirectNBody(self.w0, particle_potentials=particle_potentials_None)
        nbody = DirectNBody(
            self.w0,
            particle_potentials=self.particle_potentials,
            external_potential=self.ext_pot,
        )
        nbody = DirectNBody(
            self.w0,
            particle_potentials=self.particle_potentials,
            external_potential=self.ext_pot,
            units=usys2,
        )
        nbody = DirectNBody(self.w0, particle_potentials=[None, None], units=usys2)
        nbody = DirectNBody(
            self.w0,
            particle_potentials=[None, None],
            external_potential=self.ext_pot,
        )

        # Different INVALID ways to initialize
        with pytest.raises(TypeError):
            DirectNBody("sdf", particle_potentials=self.particle_potentials)

        with pytest.raises(ValueError):
            DirectNBody(self.w0, particle_potentials=self.particle_potentials[:1])

        with pytest.raises(ValueError):
            DirectNBody(self.w0, particle_potentials=[None, None])

    @pytest.mark.parametrize(
        "Integrator", [DOPRI853Integrator, Ruth4Integrator, LeapfrogIntegrator]
    )
    def test_directnbody_integrate(self, Integrator):
        """
        TODO: this is really a unit test, but we should have some functional tests
        that check that the orbit integration is making sense!

        Here, nbody1 has two test mass particles (massless) and nbody2 has
        one potential with mass [1] and one without [0]. This means that the orbit of
        particle [1] should be the same in both cases, but the orbit of particle [0]
        should be different (because it feels the mass of the other particle in one
        case).
        """

        atol = 1e-10 * u.pc

        # First, compare with/without mass with no external potential:
        nbody1 = DirectNBody(self.w0, particle_potentials=[None, None], units=self.usys)
        nbody2 = DirectNBody(
            self.w0, particle_potentials=self.particle_potentials, units=self.usys
        )

        orbits1 = nbody1.integrate_orbit(
            dt=1 * self.usys["time"], t1=0, t2=1 * u.Myr, Integrator=Integrator
        )
        orbits2 = nbody2.integrate_orbit(
            dt=1 * self.usys["time"], t1=0, t2=1 * u.Myr, Integrator=Integrator
        )

        dx0 = orbits1[:, 0].xyz - orbits2[:, 0].xyz
        dx1 = orbits1[:, 1].xyz - orbits2[:, 1].xyz
        assert np.abs(dx0).max() > 50 * u.pc
        assert u.allclose(np.abs(dx1), 0 * u.pc, atol=atol)

        # Now compare with/without mass with external potential:
        nbody1 = DirectNBody(
            self.w0,
            particle_potentials=[None, None],
            units=self.usys,
            external_potential=self.ext_pot,
        )
        nbody2 = DirectNBody(
            self.w0,
            particle_potentials=self.particle_potentials,
            units=self.usys,
            external_potential=self.ext_pot,
        )

        orbits1 = nbody1.integrate_orbit(
            dt=1 * self.usys["time"], t1=0, t2=1 * u.Myr, Integrator=Integrator
        )
        orbits2 = nbody2.integrate_orbit(
            dt=1 * self.usys["time"], t1=0, t2=1 * u.Myr, Integrator=Integrator
        )

        dx0 = orbits1[:, 0].xyz - orbits2[:, 0].xyz
        dx1 = orbits1[:, 1].xyz - orbits2[:, 1].xyz
        assert u.allclose(np.abs(dx1), 0 * u.pc, atol=atol)
        assert np.abs(dx0).max() > 50 * u.pc

    def test_directnbody_acceleration(self):
        pot1 = HernquistPotential(m=1e6 * u.Msun, c=0.1 * u.pc, units=self.usys)
        pot2 = HernquistPotential(m=1.6e6 * u.Msun, c=0.33 * u.pc, units=self.usys)

        nbody = DirectNBody(
            self.w0, particle_potentials=[pot1, pot2], external_potential=self.ext_pot
        )

        # Compute the acceleration we expect:
        pot1_ = HernquistPotential(
            m=1e6 * u.Msun, c=0.1 * u.pc, units=self.usys, origin=self.w0[0].xyz
        )
        pot2_ = HernquistPotential(
            m=1.6e6 * u.Msun, c=0.33 * u.pc, units=self.usys, origin=self.w0[1].xyz
        )
        exp_acc = np.zeros((3, 2)) * self.usys["acceleration"]
        exp_acc[:, 0] = pot2_.acceleration(self.w0[0])[:, 0]
        exp_acc[:, 1] = pot1_.acceleration(self.w0[1])[:, 0]
        exp_acc += self.ext_pot.acceleration(self.w0)

        acc = nbody.acceleration()
        assert u.allclose(acc, exp_acc)

    @pytest.mark.parametrize(
        "Integrator", [DOPRI853Integrator, Ruth4Integrator, LeapfrogIntegrator]
    )
    def test_directnbody_integrate_dontsaveall(self, Integrator):
        # If we set save_all = False, only return the final positions:
        nbody1 = DirectNBody(
            self.w0,
            particle_potentials=self.particle_potentials,
            units=self.usys,
            external_potential=self.ext_pot,
            save_all=False,
        )
        nbody2 = DirectNBody(
            self.w0,
            particle_potentials=self.particle_potentials,
            units=self.usys,
            external_potential=self.ext_pot,
            save_all=True,
        )

        w1 = nbody1.integrate_orbit(
            dt=1 * self.usys["time"], t1=0, t2=1 * u.Myr, Integrator=Integrator
        )
        orbits = nbody2.integrate_orbit(
            dt=1 * self.usys["time"], t1=0, t2=1 * u.Myr, Integrator=Integrator
        )
        w2 = orbits[-1]
        assert u.allclose(w1.xyz, w2.xyz)
        assert u.allclose(w1.v_xyz, w2.v_xyz)

    @pytest.mark.parametrize("Integrator", [DOPRI853Integrator])
    def test_directnbody_integrate_rotframe(self, Integrator):
        # Now compare with/without mass with external potential:
        frame = ConstantRotatingFrame(
            Omega=[0, 0, 1] * self.w0[0].v_y / self.w0[0].x, units=self.usys
        )
        nbody = DirectNBody(
            self.w0,
            particle_potentials=self.particle_potentials,
            units=self.usys,
            external_potential=self.ext_pot,
            frame=frame,
        )
        nbody2 = DirectNBody(
            self.w0,
            particle_potentials=self.particle_potentials,
            units=self.usys,
            external_potential=self.ext_pot,
        )

        orbits = nbody.integrate_orbit(
            dt=1 * self.usys["time"], t1=0, t2=1 * u.Myr, Integrator=Integrator
        )
        orbits_static = orbits.to_frame(StaticFrame(self.usys))

        orbits2 = nbody2.integrate_orbit(
            dt=1 * self.usys["time"], t1=0, t2=1 * u.Myr, Integrator=Integrator
        )

        assert u.allclose(orbits_static.xyz, orbits_static.xyz)
        assert u.allclose(orbits2.v_xyz, orbits2.v_xyz)

    @pytest.mark.parametrize("Integrator", [DOPRI853Integrator])
    def test_nbody_reorder(self, Integrator):
        N = 16
        rng = np.random.default_rng(seed=42)
        w0 = PhaseSpacePosition(
            pos=rng.normal(0, 5, size=(3, N)) * u.kpc,
            vel=rng.normal(0, 50, size=(3, N)) * u.km / u.s,
        )
        pots = [
            (
                HernquistPotential(1e9 * u.Msun, 1.0 * u.pc, units=galactic)
                if rng.uniform() > 0.5
                else None
            )
            for _ in range(N)
        ]
        sim = DirectNBody(
            w0,
            pots,
            external_potential=HernquistPotential(1e12, 10, units=galactic),
            units=galactic,
        )
        orbits = sim.integrate_orbit(dt=1.0 * u.Myr, t1=0, t2=100 * u.Myr)
        assert np.allclose(orbits.pos[0].xyz, w0.pos.xyz)
