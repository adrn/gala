"""Tests for the in-integrator SDE / diffusion kick (gala.dynamics.diffusion)."""

import astropy.units as u
import numpy as np
import pytest
from gala._cconfig import GSL_ENABLED

import gala.dynamics as gd
import gala.potential as gp
from gala.units import UnitSystem, galactic

if not GSL_ENABLED:
    pytest.skip(
        "skipping diffusion tests: they depend on GSL",
        allow_module_level=True,
    )

from gala.dynamics.diffusion import (
    ConstantDiffusion,
    GriddedDiffusion,
)


def _free_ensemble(N):
    Hn = gp.Hamiltonian(gp.NullPotential(units=galactic))
    w0 = gd.PhaseSpacePosition(
        pos=np.zeros((3, N)) * u.kpc, vel=np.zeros((3, N)) * u.kpc / u.Myr
    )
    return Hn, w0


def test_diffusion_off_matches_leapfrog():
    """D=0 (and drift=0) reproduces plain leapfrog exactly."""
    pot = gp.HernquistPotential(m=1e11, c=10, units=galactic)
    H = gp.Hamiltonian(pot)
    w0 = gd.PhaseSpacePosition(pos=[10.0, 0, 0] * u.kpc, vel=[0, 175.0, 0] * u.km / u.s)

    diff0 = ConstantDiffusion(D=np.zeros((6, 6)), basis="cartesian", units=galactic)
    o_diff = H.integrate_orbit(
        w0, dt=1.0, n_steps=500, Integrator="leapfrog", diffusion=diff0, kick_seed=1
    )
    o_plain = H.integrate_orbit(w0, dt=1.0, n_steps=500, Integrator="leapfrog")

    assert np.allclose(
        o_diff.xyz.to_value(u.kpc), o_plain.xyz.to_value(u.kpc), atol=1e-12
    )
    assert np.allclose(
        o_diff.v_xyz.to_value(u.kpc / u.Myr),
        o_plain.v_xyz.to_value(u.kpc / u.Myr),
        atol=1e-12,
    )


def test_velocity_diffusion_variance():
    """Free particle + velocity-block D: Var(v_i) grows as D_ii * T."""
    N = 4000
    Dvel = np.array([1e-4, 4e-4, 2.5e-4])
    D = np.zeros((6, 6))
    D[3, 3], D[4, 4], D[5, 5] = Dvel
    diff = ConstantDiffusion(D=D, basis="cartesian", units=galactic)

    Hn, w0 = _free_ensemble(N)
    dt, ns = 0.5, 200
    T = dt * ns
    orb = Hn.integrate_orbit(
        w0, dt=dt, n_steps=ns, Integrator="leapfrog", diffusion=diff, kick_seed=7
    )
    var = orb[-1].v_xyz.to_value(u.kpc / u.Myr).var(axis=1)
    assert np.allclose(var, Dvel * T, rtol=0.1)


def test_position_diffusion_variance():
    """Free particle at rest + position-block D: Var(x_i) grows as D_ii * T."""
    N = 4000
    Dpos = 2e-4
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = Dpos
    diff = ConstantDiffusion(D=D, basis="cartesian", units=galactic)

    Hn, w0 = _free_ensemble(N)
    dt, ns = 0.5, 200
    T = dt * ns
    orb = Hn.integrate_orbit(
        w0, dt=dt, n_steps=ns, Integrator="leapfrog", diffusion=diff, kick_seed=11
    )
    var = orb[-1].xyz.to_value(u.kpc).var(axis=1)
    assert np.allclose(var, Dpos * T, rtol=0.1)


def test_drift_is_deterministic():
    """A pure velocity drift (D=0) advances velocity as mu*T, seed-independent."""
    N = 50
    mu_v = np.array([0.01, -0.02, 0.005])  # kpc/Myr per Myr
    drift = np.concatenate([np.zeros(3), mu_v])
    diff = ConstantDiffusion(
        D=np.zeros((6, 6)), drift=drift, basis="cartesian", units=galactic
    )

    Hn, w0 = _free_ensemble(N)
    dt, ns = 1.0, 100
    T = dt * ns
    orb_a = Hn.integrate_orbit(
        w0, dt=dt, n_steps=ns, Integrator="leapfrog", diffusion=diff, kick_seed=1
    )
    orb_b = Hn.integrate_orbit(
        w0, dt=dt, n_steps=ns, Integrator="leapfrog", diffusion=diff, kick_seed=999
    )
    vf = orb_a[-1].v_xyz.to_value(u.kpc / u.Myr)
    assert np.allclose(vf, (mu_v * T)[:, None], atol=1e-10)
    # deterministic => seed-independent
    assert np.allclose(
        orb_a.v_xyz.to_value(u.kpc / u.Myr), orb_b.v_xyz.to_value(u.kpc / u.Myr)
    )


def test_cylindrical_basis_rotation():
    """Radial (v_R) velocity diffusion lands in x at phi=0 and in y at phi=90."""
    N = 4000
    D = np.zeros((6, 6))
    D[3, 3] = 1e-3
    diff = ConstantDiffusion(D=D, basis="cylindrical", units=galactic)
    Hn = gp.Hamiltonian(gp.NullPotential(units=galactic))

    for pos, expect in [([8.0, 0, 0], 0), ([0, 8.0, 0], 1)]:
        w0 = gd.PhaseSpacePosition(
            pos=np.tile(np.array(pos)[:, None], (1, N)) * u.kpc,
            vel=np.zeros((3, N)) * u.kpc / u.Myr,
        )
        orb = Hn.integrate_orbit(
            w0, dt=1.0, n_steps=1, Integrator="leapfrog", diffusion=diff, kick_seed=3
        )
        var = orb[-1].v_xyz.to_value(u.kpc / u.Myr).var(axis=1)
        # variance concentrated in the expected component
        assert var[expect] > 5e-4
        assert np.all(np.delete(var, expect) < 1e-8)


def test_full_tensor_covariance():
    """Free particle + correlated velocity tensor: covariance recovered ~ D*T."""
    N = 6000
    Dvv = np.array([[3e-4, 1e-4, 0.0], [1e-4, 2e-4, 0.0], [0.0, 0.0, 1e-4]])
    D = np.zeros((6, 6))
    D[3:, 3:] = Dvv
    diff = ConstantDiffusion(D=D, basis="cartesian", units=galactic)

    Hn, w0 = _free_ensemble(N)
    dt, ns = 0.5, 200
    T = dt * ns
    orb = Hn.integrate_orbit(
        w0, dt=dt, n_steps=ns, Integrator="leapfrog", diffusion=diff, kick_seed=5
    )
    cov = np.cov(orb[-1].v_xyz.to_value(u.kpc / u.Myr))
    target = Dvv * T
    mask = target != 0
    assert np.max(np.abs((cov - target)[mask] / target[mask])) < 0.15


def test_gridded_matches_constant():
    """A grid of constant D reproduces the ConstantDiffusion result."""
    N = 4000
    val = 1e-3
    nR, nz = 6, 6
    Rg = np.linspace(4, 12, nR)
    zg = np.linspace(0, 3, nz)
    Dgrid = np.zeros((nR, nz, 6, 6))
    Dgrid[:, :, 3, 3] = val
    gdiff = GriddedDiffusion(Rg, zg, Dgrid, basis="cylindrical", units=galactic)

    D = np.zeros((6, 6))
    D[3, 3] = val
    cdiff = ConstantDiffusion(D=D, basis="cylindrical", units=galactic)

    Hn = gp.Hamiltonian(gp.NullPotential(units=galactic))
    w0 = gd.PhaseSpacePosition(
        pos=np.tile([[8.0], [0.0], [0.0]], (1, N)) * u.kpc,
        vel=np.zeros((3, N)) * u.kpc / u.Myr,
    )
    kw = {"dt": 1.0, "n_steps": 1, "Integrator": "leapfrog", "kick_seed": 3}
    var_g = (
        Hn.integrate_orbit(w0, diffusion=gdiff, **kw)[-1]
        .v_xyz.to_value(u.kpc / u.Myr)
        .var(axis=1)
    )
    var_c = (
        Hn.integrate_orbit(w0, diffusion=cdiff, **kw)[-1]
        .v_xyz.to_value(u.kpc / u.Myr)
        .var(axis=1)
    )
    assert np.allclose(var_g, var_c)


def test_gridded_interpolates_radial_profile():
    """D_RR = a*R over the grid reproduces the expected amplitude at two radii."""
    N = 5000
    a = 1e-4  # (kpc/Myr)^2 / Myr / kpc
    nR, nz = 10, 5
    Rg = np.linspace(2.0, 14.0, nR)
    zg = np.linspace(0, 2.0, nz)
    Dgrid = np.zeros((nR, nz, 6, 6))
    Dgrid[:, :, 3, 3] = (a * Rg)[:, None]  # linear in R, flat in z
    gdiff = GriddedDiffusion(Rg, zg, Dgrid, basis="cylindrical", units=galactic)
    Hn = gp.Hamiltonian(gp.NullPotential(units=galactic))

    dt = 1.0
    for R in (6.0, 10.0):
        w0 = gd.PhaseSpacePosition(
            pos=np.tile([[R], [0.0], [0.0]], (1, N)) * u.kpc,
            vel=np.zeros((3, N)) * u.kpc / u.Myr,
        )
        orb = Hn.integrate_orbit(
            w0, dt=dt, n_steps=1, Integrator="leapfrog", diffusion=gdiff, kick_seed=2
        )
        var_x = orb[-1].v_xyz.to_value(u.kpc / u.Myr)[0].var()
        assert np.isclose(var_x, a * R * dt, rtol=0.1)


def test_reproducibility():
    """Same kick_seed -> identical; different -> different."""
    D = np.zeros((6, 6))
    D[3, 3] = D[4, 4] = D[5, 5] = 1e-4
    diff = ConstantDiffusion(D=D, basis="cartesian", units=galactic)
    Hn, w0 = _free_ensemble(20)
    kw = {"dt": 1.0, "n_steps": 50, "Integrator": "leapfrog", "diffusion": diff}
    a = Hn.integrate_orbit(w0, kick_seed=42, **kw)
    b = Hn.integrate_orbit(w0, kick_seed=42, **kw)
    c = Hn.integrate_orbit(w0, kick_seed=43, **kw)
    assert np.array_equal(a.v_xyz.value, b.v_xyz.value)
    assert not np.array_equal(a.v_xyz.value, c.v_xyz.value)


def test_unsupported_integrator_raises():
    """diffusion= with a non-leapfrog integrator or the Python path raises."""
    diff = ConstantDiffusion(D=np.zeros((6, 6)), basis="cartesian", units=galactic)
    H = gp.Hamiltonian(gp.HernquistPotential(m=1e11, c=10, units=galactic))
    w0 = gd.PhaseSpacePosition(pos=[10.0, 0, 0] * u.kpc, vel=[0, 175.0, 0] * u.km / u.s)

    with pytest.raises(NotImplementedError):
        H.integrate_orbit(w0, dt=1.0, n_steps=10, Integrator="dopri853", diffusion=diff)
    with pytest.raises(NotImplementedError):
        H.integrate_orbit(
            w0,
            dt=1.0,
            n_steps=10,
            Integrator="leapfrog",
            diffusion=diff,
            cython_if_possible=False,
        )


def test_units_mismatch_raises():
    """diffusion.units must match the Hamiltonian units."""
    other = UnitSystem(u.pc, u.Myr, u.Msun, u.radian)
    diff = ConstantDiffusion(D=np.zeros((6, 6)), basis="cartesian", units=other)
    H = gp.Hamiltonian(gp.HernquistPotential(m=1e11, c=10, units=galactic))
    w0 = gd.PhaseSpacePosition(pos=[10.0, 0, 0] * u.kpc, vel=[0, 175.0, 0] * u.km / u.s)
    with pytest.raises(ValueError):
        H.integrate_orbit(w0, dt=1.0, n_steps=10, Integrator="leapfrog", diffusion=diff)


def test_model_input_validation():
    """Bad coefficient inputs raise clear errors."""
    with pytest.raises(ValueError):  # not symmetric
        ConstantDiffusion(D=np.arange(36.0).reshape(6, 6), units=galactic)
    with pytest.raises(TypeError):  # astropy Quantity not allowed
        ConstantDiffusion(D=np.zeros((6, 6)) * u.kpc**2 / u.Myr, units=galactic)
    with pytest.raises(ValueError):  # grid too small for bicubic
        GriddedDiffusion(
            np.linspace(4, 12, 3),
            np.linspace(0, 3, 3),
            np.zeros((3, 3, 6, 6)),
            units=galactic,
        )
