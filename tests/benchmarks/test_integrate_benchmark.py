import astropy.units as u
import numpy as np
import pytest

import gala.dynamics as gd
import gala.potential as gp


@pytest.fixture
def potential():
    return gp.HernquistPotential(m=1e11, c=5, units="galactic")


def make_w0(v_y, n_orbits, n_steps):
    if n_orbits * n_steps > 1_000_000:
        pytest.skip("too slow under CodSpeed's simulation")
    xyz = np.zeros((3, n_orbits))
    xyz[0] = 10.0
    vxyz = np.zeros((3, n_orbits))
    vxyz[1] = v_y
    return gd.PhaseSpacePosition(xyz * u.kpc, vxyz * u.km / u.s)


@pytest.fixture
def circular_w0(n_orbits, n_steps):
    return make_w0(138.25768647, n_orbits, n_steps)


@pytest.fixture
def eccentric_w0(n_orbits, n_steps):
    return make_w0(0.25 * 138.25768647, n_orbits, n_steps)


@pytest.mark.benchmark
@pytest.mark.parametrize("Integrator", ["dop853", "leapfrog", "ruth4"])
@pytest.mark.parametrize("n_steps", [1, 10, 1_000, 100_000])
@pytest.mark.parametrize("n_orbits", [1, 10, 1_000, 100_000])
def test_integrate_circular_orbit(
    potential, circular_w0, Integrator, n_steps, n_orbits
):
    potential.integrate_orbit(
        circular_w0, dt=1 * u.Myr, n_steps=n_steps, Integrator=Integrator
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("Integrator", ["dop853", "leapfrog", "ruth4"])
@pytest.mark.parametrize("n_steps", [1, 10, 1_000, 100_000])
@pytest.mark.parametrize("n_orbits", [1, 10, 1_000, 100_000])
def test_integrate_eccentric_orbit(
    potential, eccentric_w0, Integrator, n_steps, n_orbits
):
    potential.integrate_orbit(
        eccentric_w0, dt=1 * u.Myr, n_steps=n_steps, Integrator=Integrator
    )
