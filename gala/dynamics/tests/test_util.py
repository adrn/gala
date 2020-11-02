# Third-party
import astropy.units as u
import numpy as np
import pytest

# Project
from ..core import PhaseSpacePosition
from ..orbit import Orbit
from ..util import peak_to_peak_period, estimate_dt_n_steps, combine
from ...potential import Hamiltonian, NFWPotential
from ...potential.frame import StaticFrame
from ...units import galactic


def test_peak_to_peak_period():
    ntimes = 16384

    # trivial test
    for true_T in [1., 2., 4.123]:
        t = np.linspace(0, 10., ntimes)
        f = np.sin(2*np.pi/true_T * t)
        T = peak_to_peak_period(t, f)
        assert np.allclose(T, true_T, atol=1E-3)

    # modulated trivial test
    true_T = 2.
    t = np.linspace(0, 10., ntimes)
    f = np.sin(2*np.pi/true_T * t) + 0.1*np.cos(2*np.pi/(10*true_T) * t)
    T = peak_to_peak_period(t, f)
    assert np.allclose(T, true_T, atol=1E-3)


def test_estimate_dt_n_steps():
    nperiods = 128
    pot = NFWPotential.from_circular_velocity(v_c=1., r_s=10., units=galactic)
    w0 = [10., 0., 0., 0., 0.9, 0.]

    H = Hamiltonian(pot)
    dt, n_steps = estimate_dt_n_steps(w0, H, n_periods=nperiods,
                                      n_steps_per_period=256,
                                      func=np.nanmin)

    orbit = H.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    T = orbit.estimate_period()
    assert int(round((orbit.t.max()/T).decompose().value)) == nperiods


class TestCombine(object):

    def setup(self):
        x = np.random.random(size=(3,))
        v = np.random.random(size=(3,))
        p1 = PhaseSpacePosition(pos=x, vel=v)
        p2 = PhaseSpacePosition(pos=x, vel=v, frame=StaticFrame(galactic))
        x = np.random.random(size=(3, 5))
        v = np.random.random(size=(3, 5))
        p3 = PhaseSpacePosition(pos=x, vel=v)
        p4 = PhaseSpacePosition(pos=x*u.kpc, vel=v*u.km/u.s)
        x = np.random.random(size=(2, 5))
        v = np.random.random(size=(2, 5))
        p5 = PhaseSpacePosition(pos=x, vel=v)
        self.psps = [p1, p2, p3, p4, p5]

        x = np.random.random(size=(3, 8))
        v = np.random.random(size=(3, 8))
        o1 = Orbit(pos=x, vel=v)
        o2 = Orbit(pos=x, vel=v, t=np.arange(8))

        pot = NFWPotential.from_circular_velocity(v_c=1., r_s=10.,
                                                  units=galactic)
        o3 = Orbit(pos=x*u.kpc, vel=v*u.km/u.s, t=np.arange(8)*u.Myr,
                   potential=pot, frame=StaticFrame(galactic))

        x = np.random.random(size=(2, 8))
        v = np.random.random(size=(2, 8))
        o4 = Orbit(pos=x, vel=v, t=np.arange(8))
        self.orbs = [o1, o2, o3, o4]

    def test_combine_fail(self):

        with pytest.raises(ValueError):
            combine([])

        with pytest.raises(ValueError):
            combine(self.psps[0])

        with pytest.raises(TypeError):
            combine([self.psps[0], self.orbs[0]])

        with pytest.raises(TypeError):
            combine([5, 5, 5])

        with pytest.raises(ValueError):
            combine(self.psps)

        with pytest.raises(ValueError):
            combine(self.orbs)

    def test_combine_psp(self):

        for psp in self.psps:
            psps = [psp] * 3
            new_psp = combine(psps)
            assert new_psp.ndim == psp.ndim

            if psp.pos.shape:
                shp = psp.pos.shape
            else:
                shp = (1,)

            assert new_psp.pos.shape == (3*shp[0],)
            assert new_psp.frame == psp.frame

    def test_combine_orb(self):

        for orb in self.orbs:
            orbs = [orb] * 4
            new_orb = combine(orbs)
            assert new_orb.ndim == orb.ndim

            shp = orb.shape
            if len(shp) < 2:
                shp = shp + (4,)

            else:
                shp = shp[:-1] + (4*shp[-1],)

            assert new_orb.pos.shape == shp
            assert new_orb.frame == orb.frame
            assert new_orb.potential == orb.potential
