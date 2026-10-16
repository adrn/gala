import sys

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import G
from gala._cconfig import GSL_ENABLED
from staeckel_helpers import galpy_find_actions_staeckel_fudge

import gala.potential as gp
from gala._optional_deps import HAS_GALPY
from gala.dynamics import PhaseSpacePosition
from gala.dynamics.actionangle import (
    find_actions_o2gf,
    find_actions_staeckel_fudge,
    get_staeckel_fudge_delta,
    isochrone_xv_to_aa,
)
from gala.units import galactic

requires_gsl = pytest.mark.skipif(not GSL_ENABLED, reason="requires GSL")
# Last I tried, galpy's actionAngleStaeckel C implementation segfaults on macOS?
requires_galpy = pytest.mark.skipif(
    not HAS_GALPY or sys.platform == "darwin",
    reason="requires galpy",
)
staeckel_finders = [
    pytest.param(find_actions_staeckel_fudge, marks=requires_gsl, id="gala"),
    pytest.param(galpy_find_actions_staeckel_fudge, marks=requires_galpy, id="galpy"),
]


@pytest.mark.skipif(not HAS_GALPY, reason="requires galpy to run this test")
def test_staeckel_fudge_delta():
    import galpy.potential as galpy_pot
    from galpy.actionAngle import estimateDeltaStaeckel

    ro = 8.1 * u.kpc
    vo = 229 * u.km / u.s

    paired_potentials = []

    # Miyamoto-Nagai
    potential = gp.MiyamotoNagaiPotential(
        m=6e10 * u.Msun, a=3 * u.kpc, b=0.3 * u.kpc, units=galactic
    )
    amp = (G * potential.parameters["m"]).to_value(vo**2 * ro)
    a = potential.parameters["a"].to_value(ro)
    b = potential.parameters["b"].to_value(ro)
    galpy_potential = galpy_pot.MiyamotoNagaiPotential(amp=amp, a=a, b=b, ro=ro, vo=vo)
    paired_potentials.append((potential, galpy_potential))

    # Hernquist
    potential = gp.HernquistPotential(m=6e10 * u.Msun, c=0.3 * u.kpc, units=galactic)
    amp = (G * potential.parameters["m"]).to_value(vo**2 * ro)
    a = potential.parameters["c"].to_value(ro)
    galpy_potential = galpy_pot.HernquistPotential(amp=amp, a=a, ro=ro, vo=vo)
    paired_potentials.append((potential, galpy_potential))

    # NFW
    potential = gp.NFWPotential(m=6e11 * u.Msun, r_s=15.6 * u.kpc, units=galactic)
    amp = (G * potential.parameters["m"]).to_value(vo**2 * ro)
    a = potential.parameters["r_s"].to_value(ro)
    galpy_potential = galpy_pot.NFWPotential(amp=amp, a=a, ro=ro, vo=vo)
    paired_potentials.append((potential, galpy_potential))

    # TEST:
    # TODO: remove the randomness here
    N = 1024
    rnd = np.random.default_rng(42)
    w = PhaseSpacePosition(
        pos=rnd.uniform(-10, 10, size=(3, N)) * u.kpc,
        vel=rnd.uniform(-100, 100, size=(3, N)) * u.km / u.s,
    )

    R = w.cylindrical.rho.to_value(ro)
    z = w.z.to_value(ro)

    for p, galpy_p in paired_potentials:
        galpy_deltas = estimateDeltaStaeckel(galpy_p, R, z, no_median=True)
        gala_deltas = get_staeckel_fudge_delta(p, w).value
        assert np.allclose(gala_deltas, galpy_deltas, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("find_actions", staeckel_finders)
def test_find_actions_staeckel_fudge(find_actions):
    """
    This test function performs some unit test checks of the API
    """
    disk = gp.MiyamotoNagaiPotential(5e10, 3.5, 0.3, units=galactic)
    halo = gp.NFWPotential.from_M200_c(1e12 * u.Msun, 15, units=galactic)
    pot = disk + halo

    xyz = (np.zeros((3, 16)) + 1e-5) * u.kpc
    xyz[0] = np.linspace(4, 20, xyz.shape[1]) * u.kpc

    vxyz = np.zeros((3, 16)) * u.km / u.s
    vxyz[0] = 15 * u.km / u.s
    vxyz[1] = pot.circular_velocity(xyz)
    vxyz[2] = 15 * u.km / u.s

    w0_one = PhaseSpacePosition(xyz[:, 0], vxyz[:, 0])
    w0_many = PhaseSpacePosition(xyz, vxyz)
    orbit_one = pot.integrate_orbit(w0_one, dt=1.0, n_steps=1000)
    orbit_many = pot.integrate_orbit(w0_many, dt=1.0, n_steps=1000)

    inputs = [w0_one, w0_many, orbit_one, orbit_many]
    shapes = [(1, 3), (xyz.shape[1], 3), (1, 3), (xyz.shape[1], 3)]
    for w, colshape in zip(inputs, shapes):
        aaf = find_actions(pot, w)

        for colname in ["actions", "freqs"]:
            assert aaf[colname].shape == colshape

    # Check that mean=False returns the right shape
    aaf = find_actions(pot, orbit_one, mean=False)
    for colname in ["actions", "freqs", "angles"]:
        assert aaf[colname].shape == (1, orbit_one.ntimes, 3)

    aaf = find_actions(pot, orbit_many, mean=False)
    for colname in ["actions", "freqs", "angles"]:
        assert aaf[colname].shape == (xyz.shape[1], orbit_one.ntimes, 3)


@pytest.mark.parametrize("find_actions", staeckel_finders)
def test_compare_staeckel_o2gf(find_actions):
    """
    This test function performs some comparisons between actions, angles,
    and frequencies solved from the staeckel fudge and O2GF.
    """
    disk = gp.MiyamotoNagaiPotential(5e10, 3.5, 0.3, units=galactic)
    halo = gp.NFWPotential.from_M200_c(1e12 * u.Msun, 15, units=galactic)
    pot = disk + halo

    xyz = (np.zeros((3, 16)) + 1e-5) * u.kpc
    xyz[0] = np.linspace(4, 20, xyz.shape[1]) * u.kpc

    vxyz = np.zeros((3, 16)) * u.km / u.s
    vxyz[0] = 15 * u.km / u.s
    vxyz[1] = pot.circular_velocity(xyz)
    vxyz[2] = 15 * u.km / u.s

    orbits = pot.integrate_orbit(PhaseSpacePosition(xyz, vxyz), dt=1.0, n_steps=20_000)

    aaf_staeckel = find_actions(pot, orbits)
    aaf_o2gf = find_actions_o2gf(orbits, N_max=10)

    assert u.allclose(aaf_staeckel["actions"], aaf_o2gf["actions"], rtol=1e-3)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        assert u.allclose(aaf_staeckel["freqs"], aaf_o2gf["freqs"], rtol=1e-3)
    assert u.allclose(aaf_staeckel["angles"], aaf_o2gf["angles"], rtol=1.5e-2)


@requires_gsl
def test_staeckel_isochrone():
    """In a spherical potential with Delta -> 0, the fudge is exact up to quadrature."""
    pot = gp.IsochronePotential(m=1e11, b=3.0, units=galactic)
    rng = np.random.default_rng(42)
    N = 256
    w = PhaseSpacePosition(
        pos=rng.uniform(-8, 8, size=(3, N)) * u.kpc,
        vel=rng.uniform(-150, 150, size=(3, N)) * u.km / u.s,
    )
    aaf = find_actions_staeckel_fudge(pot, w, delta=0.0)
    actions, angles, freqs = isochrone_xv_to_aa(w, pot)

    # nearly radial orbits are less accurate at fixed quadrature order
    for ours, truth in [(aaf["actions"], actions.T), (aaf["freqs"], freqs.T)]:
        rel = np.abs(ours - truth) / np.abs(truth)
        assert np.all(np.nanmedian(rel, axis=0) < 1e-5)
        assert np.all(np.nanpercentile(rel, 90, axis=0) < 1e-3)

    dtheta = (aaf["angles"] - angles.T).to_value(u.rad)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    assert np.all(np.nanmedian(np.abs(dtheta), axis=0) < 1e-5)


@requires_gsl
def test_staeckel_conservation():
    """Actions are nearly conserved along orbits in a realistic disk + halo potential."""
    pot = gp.MilkyWayPotential2022()
    w0 = PhaseSpacePosition(
        pos=[[8.0, 6, 10], [0, 0, 0], [0.5, -1, 2]] * u.kpc,
        vel=[[30, 60, -20], [220, 180, 200], [40, -30, 50]] * u.km / u.s,
    )
    orbits = pot.integrate_orbit(w0, dt=1.0, n_steps=4000)
    aaf = find_actions_staeckel_fudge(pot, orbits, mean=False)
    J = aaf["actions"].value
    frac_scatter = np.std(J, axis=1) / np.abs(np.mean(J, axis=1))
    # the fudge typically conserves J_R to a few percent
    assert np.all(frac_scatter[:, 0] < 5e-2)  # J_R
    assert np.all(frac_scatter[:, 1] < 1e-10)  # J_phi = L_z exactly
    assert np.all(frac_scatter[:, 2] < 1e-2)  # J_z


@requires_gsl
def test_staeckel_edge_cases():
    disk = gp.MiyamotoNagaiPotential(5e10, 3.5, 0.3, units=galactic)
    halo = gp.NFWPotential.from_M200_c(1e12 * u.Msun, 15, units=galactic)
    pot = disk + halo

    w = PhaseSpacePosition(
        pos=[[8.0, 8.0, 8.0], [0, 0, 0], [0, 1.0, 1.0]] * u.kpc,
        vel=[[30, 30, 1000.0], [200, 0, 200], [0, 30, 0]] * u.km / u.s,
    )
    aaf = find_actions_staeckel_fudge(pot, w)
    J = aaf["actions"].value
    Omega = aaf["freqs"].value

    # planar orbit: J_z = 0, other values finite
    assert J[0, 2] == 0
    assert np.all(np.isfinite(J[0]))
    assert np.all(np.isfinite(Omega[0, :2]))

    # radial orbit in the meridional plane (L_z = 0)
    assert J[1, 1] == 0
    assert np.all(np.isfinite(J[1]))
    assert np.all(np.isfinite(Omega[1]))

    # unbound: NaN except J_phi
    assert np.all(np.isnan(J[2, [0, 2]]))
    assert np.isfinite(J[2, 1])

    bar = gp.LongMuraliBarPotential(m=1e10, a=4, b=1, c=1, units=galactic)
    w_bar = PhaseSpacePosition(
        pos=[4.0, 3.0, 0.5] * u.kpc, vel=[0, 150, 10] * u.km / u.s
    )
    with pytest.raises(ValueError, match="axisymmetric"):
        find_actions_staeckel_fudge(bar, w_bar)


@requires_gsl
def test_compare_staeckel_agama():
    """The implementation is a port of Agama's direct Staeckel fudge."""
    agama = pytest.importorskip("agama")
    G_ = G.decompose(galactic).value
    pot = gp.CCompositePotential(
        disk=gp.MiyamotoNagaiPotential(m=6.8e10, a=3.0, b=0.28, units=galactic),
        halo=gp.NFWPotential(m=5.4e11, r_s=15.6, units=galactic),
    )
    agama_pot = agama.Potential(
        {
            "type": "MiyamotoNagai",
            "mass": G_ * 6.8e10,
            "scaleRadius": 3.0,
            "scaleHeight": 0.28,
        },
        {"type": "NFW", "mass": G_ * 5.4e11, "scaleRadius": 15.6},
    )

    rng = np.random.default_rng(3)
    N = 256
    R = rng.uniform(3, 15, N)
    phi = rng.uniform(0, 2 * np.pi, N)
    vR, vphi, vz = rng.normal([0, 0.2, 0], [0.04, 0.03, 0.03], size=(N, 3)).T
    w = np.stack(
        [
            R * np.cos(phi),
            R * np.sin(phi),
            rng.normal(0, 1, N),
            vR * np.cos(phi) - vphi * np.sin(phi),
            vR * np.sin(phi) + vphi * np.cos(phi),
            vz,
        ]
    )
    aaf = find_actions_staeckel_fudge(
        pot,
        PhaseSpacePosition(pos=w[:3] * u.kpc, vel=w[3:] * u.kpc / u.Myr),
        delta=3.0 * u.kpc,
    )
    act, ang, freq = agama.actions(
        agama_pot, np.ascontiguousarray(w.T), fd=3.0, angles=True
    )

    # Agama orders (R, z, phi)
    idx = [0, 2, 1]
    assert np.allclose(aaf["actions"].value, act[:, idx], rtol=1e-6)
    assert np.allclose(aaf["freqs"].value, freq[:, idx], rtol=1e-4)
    dtheta = aaf["angles"].value - ang[:, idx]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    assert np.all(np.abs(dtheta) < 1e-3)
