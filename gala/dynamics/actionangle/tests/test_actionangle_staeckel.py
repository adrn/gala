# Third-party
from astropy.constants import G
import astropy.units as u
import numpy as np
import pytest

# gala
from gala.dynamics import PhaseSpacePosition
from gala.dynamics.actionangle import (
    get_staeckel_fudge_delta,
    find_actions_o2gf
)
from gala.dynamics.actionangle.tests.staeckel_helpers import galpy_find_actions_staeckel
import gala.potential as gp
from gala.units import galactic
from gala.tests.optional_deps import HAS_GALPY


@pytest.mark.skipif(not HAS_GALPY,
                    reason="requires galpy to run this test")
def test_staeckel_fudge_delta():
    import galpy.potential as galpy_pot
    from galpy.actionAngle import estimateDeltaStaeckel

    ro = 8.1 * u.kpc
    vo = 229 * u.km/u.s

    paired_potentials = []

    # Miyamoto-Nagai
    potential = gp.MiyamotoNagaiPotential(
        m=6e10*u.Msun, a=3*u.kpc, b=0.3*u.kpc, units=galactic)
    amp = (G * potential.parameters['m']).to_value(vo**2 * ro)
    a = potential.parameters['a'].to_value(ro)
    b = potential.parameters['b'].to_value(ro)
    galpy_potential = galpy_pot.MiyamotoNagaiPotential(amp=amp, a=a, b=b,
                                                       ro=ro, vo=vo)
    paired_potentials.append((potential, galpy_potential))

    # Hernquist
    potential = gp.HernquistPotential(m=6e10*u.Msun, c=0.3*u.kpc,
                                      units=galactic)
    amp = (G * potential.parameters['m']).to_value(vo**2 * ro)
    a = potential.parameters['c'].to_value(ro)
    galpy_potential = galpy_pot.HernquistPotential(amp=amp, a=a,
                                                   ro=ro, vo=vo)
    paired_potentials.append((potential, galpy_potential))

    # NFW
    potential = gp.NFWPotential(m=6e11*u.Msun, r_s=15.6*u.kpc,
                                units=galactic)
    amp = (G * potential.parameters['m']).to_value(vo**2 * ro)
    a = potential.parameters['r_s'].to_value(ro)
    galpy_potential = galpy_pot.NFWPotential(amp=amp, a=a, ro=ro, vo=vo)
    paired_potentials.append((potential, galpy_potential))

    # TEST:
    # TODO: remove the randomness here
    N = 1024
    rnd = np.random.default_rng(42)
    w = PhaseSpacePosition(
        pos=rnd.uniform(-10, 10, size=(3, N)) * u.kpc,
        vel=rnd.uniform(-100, 100, size=(3, N)) * u.km/u.s
    )

    R = w.cylindrical.rho.to_value(ro)
    z = w.z.to_value(ro)

    for p, galpy_p in paired_potentials:
        galpy_deltas = estimateDeltaStaeckel(galpy_p, R, z,
                                             no_median=True)
        gala_deltas = get_staeckel_fudge_delta(p, w).value
        assert np.allclose(gala_deltas, galpy_deltas, atol=1e-5, rtol=1e-3)


@pytest.mark.skipif(not HAS_GALPY,
                    reason="requires galpy to run this test")
def test_find_actions_staeckel():
    """
    This test function performs some unit test checks of the API
    """
    disk = gp.MiyamotoNagaiPotential(5e10, 3.5, 0.3, units=galactic)
    halo = gp.NFWPotential.from_M200_c(1e12*u.Msun, 15, units=galactic)
    pot = disk + halo

    xyz = (np.zeros((3, 16)) + 1e-5) * u.kpc
    xyz[0] = np.linspace(4, 20, xyz.shape[1]) * u.kpc

    vxyz = np.zeros((3, 16)) * u.km/u.s
    vxyz[0] = 15 * u.km/u.s
    vxyz[1] = pot.circular_velocity(xyz)
    vxyz[2] = 15 * u.km/u.s

    w0_one = PhaseSpacePosition(xyz[:, 0], vxyz[:, 0])
    w0_many = PhaseSpacePosition(xyz, vxyz)
    orbit_one = pot.integrate_orbit(w0_one, dt=1., n_steps=1000)
    orbit_many = pot.integrate_orbit(w0_many, dt=1., n_steps=1000)

    inputs = [
        w0_one,
        w0_many,
        orbit_one,
        orbit_many
    ]
    shapes = [
        (1, 3),
        (xyz.shape[1], 3),
        (1, 3),
        (xyz.shape[1], 3)
    ]
    for w, colshape in zip(inputs, shapes):
        aaf = galpy_find_actions_staeckel(pot, w)

        for colname in ['actions', 'freqs']:
            assert aaf[colname].shape == colshape

    # Check that mean=False returns the right shape
    aaf = galpy_find_actions_staeckel(pot, orbit_one, mean=False)
    for colname in ['actions', 'freqs', 'angles']:
        assert aaf[colname].shape == (1, orbit_one.ntimes, 3)

    aaf = galpy_find_actions_staeckel(pot, orbit_many, mean=False)
    for colname in ['actions', 'freqs', 'angles']:
        assert aaf[colname].shape == (xyz.shape[1], orbit_one.ntimes, 3)


@pytest.mark.skipif(not HAS_GALPY,
                    reason="requires galpy to run this test")
def test_compare_staeckel_o2gf():
    """
    This test function performs some comparisons between actions, angles,
    and frequencies solved from the staeckel fudge and O2GF.
    """
    disk = gp.MiyamotoNagaiPotential(5e10, 3.5, 0.3, units=galactic)
    halo = gp.NFWPotential.from_M200_c(1e12*u.Msun, 15, units=galactic)
    pot = disk + halo

    xyz = (np.zeros((3, 16)) + 1e-5) * u.kpc
    xyz[0] = np.linspace(4, 20, xyz.shape[1]) * u.kpc

    vxyz = np.zeros((3, 16)) * u.km/u.s
    vxyz[0] = 15 * u.km/u.s
    vxyz[1] = pot.circular_velocity(xyz)
    vxyz[2] = 15 * u.km/u.s

    orbits = pot.integrate_orbit(
        PhaseSpacePosition(xyz, vxyz),
        dt=1., n_steps=20_000
    )

    aaf_staeckel = galpy_find_actions_staeckel(pot, orbits)
    aaf_o2gf = find_actions_o2gf(orbits, N_max=10)

    assert u.allclose(aaf_staeckel['actions'], aaf_o2gf['actions'], rtol=1e-3)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        assert u.allclose(aaf_staeckel['freqs'], aaf_o2gf['freqs'], rtol=1e-3)
    assert u.allclose(aaf_staeckel['angles'], aaf_o2gf['angles'], rtol=1.5e-2)
