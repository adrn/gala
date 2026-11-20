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


@pytest.mark.parametrize("kick_every", [1, 7])
def test_velocity_diffusion_variance(kick_every):
    """Free particle + velocity-block D: Var(v_i) grows as D_ii * T.

    kick_every=7 does not divide the 200 steps, so this also checks that the final
    partial interval is kicked with its true length.
    """
    N = 4000
    Dvel = np.array([1e-4, 4e-4, 2.5e-4])
    D = np.zeros((6, 6))
    D[3, 3], D[4, 4], D[5, 5] = Dvel
    diff = ConstantDiffusion(
        D=D, basis="cartesian", units=galactic, kick_every=kick_every
    )

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


def test_cylindrical_kick_is_curvilinear():
    """An azimuthal (phi_hat) kick moves the star along its circle and keeps its
    velocity in the local frame, so it must not create radial velocity (rotating the
    increment into a fixed Cartesian frame would add v_phi * dphi to v_R)."""
    D = np.zeros((6, 6))
    D[1, 1] = 1.0  # kpc^2/Myr: large azimuthal position diffusion
    diff = ConstantDiffusion(D=D, basis="cylindrical", units=galactic)
    N = 500
    w0 = gd.PhaseSpacePosition(
        pos=np.tile([[8.0], [0.0], [0.0]], N) * u.kpc,
        vel=np.tile([[0.0], [0.2], [0.0]], N) * u.kpc / u.Myr,
    )
    H = gp.Hamiltonian(gp.NullPotential(units=galactic))
    orbit = H.integrate_orbit(w0, dt=0.1, n_steps=1, diffusion=diff, kick_seed=1)
    x, v = orbit.xyz[:, -1].to_value(u.kpc), orbit.v_xyz[:, -1].to_value(u.kpc / u.Myr)
    R = np.hypot(x[0], x[1])
    vR = (x[0] * v[0] + x[1] * v[1]) / R
    vphi = (x[0] * v[1] - x[1] * v[0]) / R
    assert np.std(x[1]) > 0.1  # kicked along the circle
    # identical across particles (only the deterministic free-flight part remains);
    # a Cartesian-frame kick would give std(v_R) ~ v_phi sqrt(D dt) / R ~ 8e-3
    assert np.std(vR) < 1e-12
    assert np.std(vphi) < 1e-12


def test_too_few_times_raises():
    """A time grid with fewer than two times must raise, not read out of bounds."""
    diff = ConstantDiffusion(D=np.eye(6) * 1e-6, basis="cartesian", units=galactic)
    H = gp.Hamiltonian(gp.NullPotential(units=galactic))
    w0 = gd.PhaseSpacePosition(pos=[1.0, 0, 0] * u.kpc, vel=[0, 0.1, 0] * u.kpc / u.Myr)
    for kw in ({}, {"diffusion": diff}):
        with pytest.raises(ValueError, match="at least two times"):
            H.integrate_orbit(w0, t=np.array([0.0]), **kw)


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


def test_gridded_out_of_bounds_is_zero():
    """Outside the (R, |z|) grid the gridded model applies no kick."""
    N = 2000
    nR, nz = 6, 5
    Rg = np.linspace(4.0, 10.0, nR)
    zg = np.linspace(0.0, 2.0, nz)
    Dgrid = np.zeros((nR, nz, 6, 6))
    Dgrid[:, :, 3, 3] = 1e-3
    gdiff = GriddedDiffusion(Rg, zg, Dgrid, basis="cylindrical", units=galactic)
    Hn = gp.Hamiltonian(gp.NullPotential(units=galactic))

    def _run(pos, n_steps):
        w0 = gd.PhaseSpacePosition(
            pos=np.tile(np.array(pos)[:, None], (1, N)) * u.kpc,
            vel=np.zeros((3, N)) * u.kpc / u.Myr,
        )
        return Hn.integrate_orbit(
            w0,
            dt=1.0,
            n_steps=n_steps,
            Integrator="leapfrog",
            diffusion=gdiff,
            kick_seed=1,
        )

    # inside the grid: nonzero diffusion
    var_in = _run([7.0, 0.0, 0.0], 1)[-1].v_xyz.to_value(u.kpc / u.Myr).var(axis=1)
    assert var_in[0] > 5e-4

    # outside in R, and outside in |z|: exactly zero kicks (velocity stays 0)
    for pos in ([20.0, 0.0, 0.0], [7.0, 0.0, 10.0]):
        orb = _run(pos, 5)
        assert np.all(orb.v_xyz.to_value(u.kpc / u.Myr) == 0.0)


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
    with pytest.raises(ValueError):  # kick cadence must be >= 1 step
        ConstantDiffusion(D=np.zeros((6, 6)), units=galactic, kick_every=0)
    with pytest.raises(TypeError):  # ... and an integer
        ConstantDiffusion(D=np.zeros((6, 6)), units=galactic, kick_every=2.5)


# ---- regression tests for review findings ----


def _vel_diffusion(scale=1e-4):
    D = np.zeros((6, 6))
    D[3, 3] = D[4, 4] = D[5, 5] = scale
    return ConstantDiffusion(D=D, basis="cartesian", units=galactic)


def test_unseeded_calls_differ():
    """kick_seed=None (default) draws fresh noise on every call."""
    Hn, w0 = _free_ensemble(20)
    kw = {"dt": 1.0, "n_steps": 20, "diffusion": _vel_diffusion()}
    a = Hn.integrate_orbit(w0, **kw)
    b = Hn.integrate_orbit(w0, **kw)
    assert not np.array_equal(a.v_xyz.value, b.v_xyz.value)


def test_gsl_aliased_seeds_differ():
    """GSL's mt19937 maps seed 0 to 4357; user seeds must stay distinct."""
    Hn, w0 = _free_ensemble(5)
    kw = {"dt": 1.0, "n_steps": 5, "diffusion": _vel_diffusion()}
    a = Hn.integrate_orbit(w0, kick_seed=0, **kw)
    b = Hn.integrate_orbit(w0, kick_seed=4357, **kw)
    assert not np.array_equal(a.v_xyz.value, b.v_xyz.value)


def test_backward_integration_raises():
    Hn, w0 = _free_ensemble(5)
    with pytest.raises(ValueError, match="forward"):
        Hn.integrate_orbit(w0, dt=-1.0, n_steps=5, diffusion=_vel_diffusion())


def test_time_interpolated_potential_accepts_diffusion():
    knots = np.array([0.0, 50.0, 100.0]) * u.Myr
    tip = gp.TimeInterpolatedPotential(
        gp.HernquistPotential,
        knots,
        m=np.array([1e11, 1.1e11, 1.2e11]) * u.Msun,
        c=10 * u.kpc,
        units=galactic,
    )
    w0 = gd.PhaseSpacePosition(pos=[10.0, 0, 0] * u.kpc, vel=[0, 175.0, 0] * u.km / u.s)
    orbit = tip.integrate_orbit(
        w0, dt=1.0, n_steps=5, diffusion=_vel_diffusion(), kick_seed=1
    )
    assert np.all(np.isfinite(orbit.xyz.value))


def test_equivalent_unit_system_accepted():
    """Only the length and time units need to match the Hamiltonian's."""
    units = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
    diff = ConstantDiffusion(D=np.zeros((6, 6)), basis="cartesian", units=units)
    Hn, w0 = _free_ensemble(2)
    Hn.integrate_orbit(w0, dt=1.0, n_steps=2, diffusion=diff)


def test_small_or_indefinite_D_rejected():
    D = np.zeros((6, 6))
    D[3, 3] = D[4, 4] = 1e-6
    D[3, 4] = 9e-10  # upper triangle only: asymmetric at a tiny absolute scale
    with pytest.raises(ValueError, match="symmetric"):
        ConstantDiffusion(D=D, units=galactic)

    D[4, 3] = 9e-10
    ConstantDiffusion(D=D, units=galactic)  # now symmetric and PSD

    D = np.zeros((6, 6))
    D[3, 3] = D[4, 4] = 1e-4
    D[3, 4] = D[4, 3] = 5e-4  # indefinite
    with pytest.raises(ValueError, match="semi-definite"):
        ConstantDiffusion(D=D, units=galactic)

    D_grid = np.zeros((4, 4, 6, 6))
    D_grid[..., 3, 3] = D_grid[..., 4, 4] = 1e-4
    D_grid[1, 2, 3, 4] = D_grid[1, 2, 4, 3] = 5e-4
    with pytest.raises(ValueError, match="semi-definite"):
        GriddedDiffusion(np.linspace(4, 12, 4), np.linspace(0, 3, 4), D_grid)


def test_gridded_input_validation():
    R, D_grid = np.linspace(4, 12, 4), np.zeros((4, 4, 6, 6))
    with pytest.raises(ValueError, match=">= 0"):
        GriddedDiffusion(R, np.linspace(-3, 3, 4), D_grid)
    with pytest.raises(ValueError, match="finite"):
        GriddedDiffusion(np.array([4.0, 6, np.nan, 12]), np.linspace(0, 3, 4), D_grid)


def test_coefficients_accessor():
    """coefficients() evaluates the model's drift and tensor at given points."""
    D = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * 1e-4
    drift = np.arange(6.0) * 1e-3
    diff = ConstantDiffusion(D=D, drift=drift, basis="cartesian", units=galactic)
    xv = np.ones((6, 3))
    mu, D_out = diff.coefficients(xv)
    np.testing.assert_allclose(mu, np.tile(drift, (3, 1)))
    np.testing.assert_allclose(D_out, np.tile(D, (3, 1, 1)), rtol=1e-12)


def test_kick_capsule_keeps_model_alive():
    import gc

    from gala.integrate.cyintegrators import leapfrog_integrate_hamiltonian

    Hn, _ = _free_ensemble(1)
    cap = _vel_diffusion().kick_capsule(1)
    gc.collect()
    t, w = leapfrog_integrate_hamiltonian(
        Hn, np.zeros((6, 3)), np.arange(10.0), kick_capsule=cap
    )
    assert np.all(np.isfinite(w))
    assert np.any(w[3:] != 0)


def test_direct_leapfrog_rejects_backward_kick():
    """The forward-only check lives in leapfrog, so direct callers are covered too."""
    from gala.integrate.cyintegrators import leapfrog_integrate_hamiltonian

    Hn, _ = _free_ensemble(1)
    cap = _vel_diffusion().kick_capsule(1)
    with pytest.raises(ValueError, match="forward"):
        leapfrog_integrate_hamiltonian(
            Hn, np.zeros((6, 3)), -np.arange(5.0), kick_capsule=cap
        )


def test_position_kick_restarts_leapfrog():
    """After a kick that moves positions, leapfrog continues exactly as if restarted
    from the kicked state (the half-step velocity uses the gradient at the new
    position, not the old one)."""
    H = gp.Hamiltonian(gp.HernquistPotential(m=1e11, c=5.0, units=galactic))
    drift = np.array([0.01, -0.02, 0.005, 0, 0, 0])  # kpc/Myr, deterministic
    diff = ConstantDiffusion(
        np.zeros((6, 6)), drift=drift, basis="cartesian", units=galactic, kick_every=3
    )
    w0 = gd.PhaseSpacePosition(
        pos=[8.0, 0, 0.5] * u.kpc, vel=[0, 0.2, 0.01] * u.kpc / u.Myr
    )
    full = H.integrate_orbit(w0, dt=1.0, n_steps=6, diffusion=diff)

    w3 = H.integrate_orbit(w0, dt=1.0, n_steps=3, diffusion=diff)[-1]
    plain = H.integrate_orbit(w3, dt=1.0, t1=3.0, n_steps=3)[-1]
    expected = plain.w(galactic)[:, 0] + drift * 3.0  # the kick at step 6
    np.testing.assert_allclose(full[-1].w(galactic)[:, 0], expected, rtol=1e-12)


def test_cylindrical_azimuthal_position_kick_keeps_R():
    """basis='cylindrical' is an SDE in the coordinates: a phi-hat kick moves the star
    along its circle, so R is unchanged (no curvature drift)."""
    D = np.zeros((6, 6))
    D[1, 1] = 1.0
    diff = ConstantDiffusion(D=D, basis="cylindrical", units=galactic)
    Hn, _ = _free_ensemble(1)
    w0 = gd.PhaseSpacePosition(
        pos=np.tile([[8.0], [0], [0]], 200) * u.kpc,
        vel=np.zeros((3, 200)) * u.kpc / u.Myr,
    )
    orb = Hn.integrate_orbit(w0, dt=0.1, n_steps=5, diffusion=diff, kick_seed=3)
    R = np.hypot(*orb.xyz[:2, -1].to_value(u.kpc))
    assert np.std(orb.xyz[1, -1].to_value(u.kpc)) > 0.1
    np.testing.assert_allclose(R, 8.0, rtol=1e-12)


def test_gridded_reflects_odd_components_below_midplane():
    """The grid is in |z|: below the plane, mu_z, mu_vz and tensor entries pairing
    one of (z, v_z) with an in-plane component flip sign."""
    R, z = np.linspace(4, 12, 4), np.linspace(0, 3, 4)
    D = np.diag([1.0, 1, 1, 1, 1, 1]) * 1e-4
    D[3, 5] = D[5, 3] = 3e-5  # v_R-v_z tilt: odd
    D[2, 5] = D[5, 2] = 2e-5  # z-v_z: even
    drift = np.array([1.0, 2, 3, 4, 5, 6]) * 1e-3
    gdiff = GriddedDiffusion(
        R, z, np.broadcast_to(D, (4, 4, 6, 6)), np.broadcast_to(drift, (4, 4, 6))
    )
    (mu_up, mu_dn), (D_up, D_dn) = gdiff.coefficients(
        np.array([[8.0, 8], [0, 0], [1, -1], [0, 0], [0, 0], [0, 0]])
    )
    np.testing.assert_allclose(mu_up, drift)
    np.testing.assert_allclose(mu_dn, drift * [1, 1, -1, 1, 1, -1])
    np.testing.assert_allclose(D_up, D, atol=1e-15)
    sign = np.ones(6)
    sign[[2, 5]] = -1
    np.testing.assert_allclose(D_dn, D * np.outer(sign, sign), atol=1e-15)
    assert D_dn[3, 5] < 0
    assert D_dn[2, 5] > 0


def test_zero_diagonal_with_offdiagonal_rejected():
    D = np.zeros((6, 6))
    D[3, 3] = 1e-8
    D[0, 3] = D[3, 0] = 1e-12  # D[0, 0] = 0, so this is indefinite
    with pytest.raises(ValueError, match="semi-definite"):
        ConstantDiffusion(D=D, units=galactic)


def test_coefficients_rejects_wrong_shape():
    diff = _vel_diffusion()
    with pytest.raises(ValueError, match="shape"):
        diff.coefficients(np.ones((2, 6)))
    mu, D = diff.coefficients(np.ones(6))  # a single point is fine
    assert mu.shape == (1, 6)
    assert D.shape == (1, 6, 6)


def test_model_does_not_alias_inputs():
    D = np.diag([0, 0, 0, 1e-4, 1e-4, 1e-4])
    diff = ConstantDiffusion(D=D, basis="cartesian", units=galactic)
    D[3, 3] = 5.0
    assert diff.D[3, 3] == 1e-4


def test_uninitialized_wrapper_raises():
    from gala.dynamics.diffusion.cydiffusion import CDiffusionWrapper

    with pytest.raises(RuntimeError, match="not initialized"):
        CDiffusionWrapper().coefficients(np.zeros((1, 6)))
    with pytest.raises(RuntimeError, match="not initialized"):
        CDiffusionWrapper().kick_capsule(1)


@pytest.mark.parametrize("kind", ["constant", "gridded"])
def test_models_pickle(kind):
    import pickle

    if kind == "constant":
        diff = ConstantDiffusion(
            np.diag([0, 0, 0, 1e-4, 2e-4, 3e-4]),
            drift=np.arange(6.0) * 1e-4,
            units=galactic,
            kick_every=2,
        )
    else:
        Dg = np.zeros((4, 4, 6, 6))
        Dg[..., 3, 3] = Dg[..., 5, 5] = 1e-4
        diff = GriddedDiffusion(
            np.linspace(4, 12, 4),
            np.linspace(0, 3, 4),
            Dg,
            units=galactic,
            kick_every=2,
        )
    diff2 = pickle.loads(pickle.dumps(diff))
    assert type(diff2) is type(diff)
    assert diff2.kick_every == 2

    H = gp.Hamiltonian(gp.HernquistPotential(m=1e11, c=5.0, units=galactic))
    w0 = gd.PhaseSpacePosition(
        pos=[8.0, 0, 0.5] * u.kpc, vel=[0, 0.2, 0] * u.kpc / u.Myr
    )
    kw = {"dt": 1.0, "n_steps": 10, "kick_seed": 4}
    a = H.integrate_orbit(w0, diffusion=diff, **kw)
    b = H.integrate_orbit(w0, diffusion=diff2, **kw)
    np.testing.assert_array_equal(a.xyz.value, b.xyz.value)
