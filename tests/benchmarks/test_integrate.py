import astropy.units as u
import pytest

import gala.dynamics as gd
import gala.potential as gp


@pytest.fixture
def potential():
    return gp.HernquistPotential(m=1e11, c=5, units="galactic")


@pytest.mark.benchmark
@pytest.mark.parametrize("Integrator", ["dop853", "leapfrog", "ruth4"])
@pytest.mark.parametrize("n_steps", [1, 10, 1_000, 100_000])
@pytest.mark.parametrize("n_orbits", [1, 10, 1_000, 100_000])
def test_integrate_circular_orbit(potential, Integrator, n_steps, n_orbits):
    w0 = gd.PhaseSpacePosition([10.0, 0, 0] * u.kpc, [0, 138.25768647, 0] * u.km / u.s)
    potential.integrate_orbit(w0, dt=1 * u.Myr, n_steps=n_steps, Integrator=Integrator)


@pytest.mark.benchmark
@pytest.mark.parametrize("Integrator", ["dop853", "leapfrog", "ruth4"])
@pytest.mark.parametrize("n_steps", [1, 10, 1_000, 100_000])
@pytest.mark.parametrize("n_orbits", [1, 10, 1_000, 100_000])
def test_integrate_eccentric_orbit(potential, Integrator, n_steps, n_orbits):
    w0 = gd.PhaseSpacePosition(
        [10.0, 0, 0] * u.kpc, [0, 0.25 * 138.25768647, 0] * u.km / u.s
    )
    potential.integrate_orbit(w0, dt=1 * u.Myr, n_steps=n_steps, Integrator=Integrator)
