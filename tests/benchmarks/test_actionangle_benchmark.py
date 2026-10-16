import astropy.units as u
import numpy as np
import pytest

import gala.dynamics as gd
import gala.potential as gp
from gala.dynamics.actionangle import find_actions_o2gf, find_actions_staeckel_fudge
from gala.units import galactic


@pytest.fixture
def potential():
    disk = gp.MiyamotoNagaiPotential(5e10, 3.5, 0.3, units=galactic)
    halo = gp.NFWPotential.from_M200_c(1e12 * u.Msun, 15, units=galactic)
    return disk + halo


def make_w0(potential, n):
    """Mildly eccentric, bound disk orbits."""
    rng = np.random.default_rng(42)
    xyz = np.zeros((3, n)) * u.kpc
    xyz[0] = rng.uniform(4, 20, n) * u.kpc
    xyz[2] = rng.normal(0, 0.1, n) * u.kpc

    vxyz = rng.normal(0, 15, (3, n)) * u.km / u.s
    vxyz[1] += potential.circular_velocity(xyz)
    return gd.PhaseSpacePosition(xyz, vxyz)


@pytest.mark.parametrize("n_points", [1, 100, 10_000, 100_000])
def test_staeckel_fudge(potential, benchmark, n_points):
    w0 = make_w0(potential, n_points)
    benchmark(find_actions_staeckel_fudge, potential, w0)


@pytest.mark.parametrize("n_steps", [1_000, 10_000])
@pytest.mark.parametrize("n_orbits", [1, 10])
def test_o2gf(potential, benchmark, n_orbits, n_steps):
    # integrate outside the timed region: integration has its own benchmarks
    orbits = potential.integrate_orbit(
        make_w0(potential, n_orbits), dt=1 * u.Myr, n_steps=n_steps
    )
    benchmark(find_actions_o2gf, orbits, N_max=8)
