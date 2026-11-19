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


# ---- ISMDiffusionWinter27 ----


def _winter27_epicyclic(pot, Lz, units):
    """Epicyclic quantities at the guiding radius for L_z, straight from the potential."""
    from scipy.optimize import brentq

    def freqs(R):
        q = np.array([[R], [0.0], [0.0]]) * units["length"]
        PR = pot.gradient(q).decompose(units).value[0, 0]
        h = pot.hessian(q).decompose(units).value[..., 0]
        return np.sqrt(PR / R), np.sqrt(h[0, 0] + 3 * PR / R), np.sqrt(h[2, 2])

    Rg = brentq(lambda R: R**2 * freqs(R)[0] - Lz, 0.5, 50.0, xtol=1e-14)
    Om, ka, nu = freqs(Rg)
    h = 1e-3  # finite differences in ln R
    qR = -(np.log(freqs(Rg * np.exp(h))[1]) - np.log(freqs(Rg * np.exp(-h))[1])) / (
        2 * h
    )
    return Rg, Om, ka, nu, qR


def _winter27_reference(
    pot,
    xv,
    units,
    J_floor_val,
    planar_suppression=True,
    environmental_scaling=False,
    long_wavelength_only=False,
    vertical=True,
):
    """Independent numpy transcription of the reference script (eom_model.py) and
    Winter-Granic et al. App. B, in physical units (kpc, km/s, Gyr)."""
    kms, kpc, Gyr = u.km / u.s, u.kpc, u.Gyr
    act = (1 * kpc * kms).decompose(units).value
    vel = (1 * kms).decompose(units).value
    freq = (1 * kms / kpc).decompose(units).value

    x, y, z, vx, vy, vz = xv
    R = np.hypot(x, y)
    vR = (x * vx + y * vy) / R / vel
    vT = (x * vy - y * vx) / R / vel
    z = z / (1 * kpc).decompose(units).value
    vz = vz / vel
    R = R / (1 * kpc).decompose(units).value
    L = R * vT
    Rg, Om, ka, nu, qR = _winter27_epicyclic(pot, L * act, units)
    Om, ka, nu = Om / freq, ka / freq, nu / freq
    Rg = Rg / (1 * kpc).decompose(units).value
    vc = Rg * Om
    floor = J_floor_val / act
    J = max((vR**2 + ka**2 * (R - Rg) ** 2) / (2 * ka), floor)
    Jz = max((vz**2 + nu**2 * z**2) / (2 * nu), floor)

    # environment: the script uses the potential's kappa, nu at 8 kpc when off
    R8 = 8.0
    q8 = np.array([[R8], [0.0], [0.0]]) * u.kpc
    PR = pot.gradient(q8).decompose(units).value[0, 0]
    h8 = pot.hessian(q8).decompose(units).value[..., 0]
    kappa8 = np.sqrt(h8[0, 0] + 3 * PR / (R8 * (1 * kpc).decompose(units).value)) / freq
    nu8 = np.sqrt(h8[2, 2]) / freq
    R_env, ka_env, nu_env = (Rg, ka, nu) if environmental_scaling else (R8, kappa8, nu8)

    big = np.inf if long_wavelength_only else None
    JR_break = big or 2.4
    Jp_break = big or 0.72
    JRz_break = big or 1.5
    Jz_break = 2.3

    kr = ka_env / (np.sqrt(2) * 28)
    rr = R_env / 8.0
    j = J
    use_high_R, use_high_phi = j > JR_break, j > Jp_break
    D_R_high = 35.0 * j**-0.5 * kr**0.5
    D_JRJR = D_R_high if use_high_R else 9.3 * j / kr
    dD = -0.5 * D_R_high / j if use_high_R else 9.3 / kr
    D_JpJp = 5900.0 * j**-0.5 * rr**2 * kr**0.5 if use_high_phi else 6900.0 * rr**2
    A_JR = 0.5 * dD
    A_Jp = 0.0
    if planar_suppression:
        Hgas, kstar = 0.25, 10.0  # kpc, 1/kpc
        KH = kstar * Hgas
        Jvert = max(2 * Jz / (nu_env * Hgas**2), 0.0)  # (kpc km/s)/(km/s/kpc kpc^2)
        pre = (1 + 0.71 * KH) ** 2
        xi = pre / (1 + (pre - 1) * (1 + KH ** (1 / 3) * Jvert**0.75))
        sR = xi / (0.75 if use_high_R else 0.94)
        A_JR *= sR
        D_JRJR *= sR
        D_JpJp *= xi / (0.84 if use_high_phi else 0.98)

    g = 2 * Om / ka
    q = 2 - 2 / g**2
    C = A_JR / (2 * J) - D_JRJR / (4 * J**2)  # 1/Gyr
    AR = (
        g**2 * Rg / (2 * L) * A_Jp
        + (R - Rg) * C
        + g**4 * Rg / (8 * L**2) * (2 * qR - q - 1) * D_JpJp
    )
    AvR = vR * C
    Aphi = g * vR / (Rg * ka) * C
    Drr = (g * L / (4 * J) * D_JRJR + g**4 / 4 * D_JpJp) / vc**2
    DrvT = -g / (4 * J) * D_JRJR + g**2 / (2 * L) * (1 - g**2 / 2) * D_JpJp
    DvTvT = vc**2 * (g / (4 * J * L) * D_JRJR + (1 - g**2 / 2) ** 2 * D_JpJp / L**2)
    DvRvR = vc**2 / g * D_JRJR / (J * L)
    DphivR = g / 2 * vc * D_JRJR / (J * L)
    Dphph = g**3 / 4 * D_JRJR / (J * L)
    AvT = (A_Jp - vT * AR - DrvT) / R

    Dzz = Dvzvz = Az = Avz = 0.0
    if vertical:
        nu_ratio = (nu_env * kms / kpc).to_value(1 / u.Myr) / (2 * np.pi / 75.55)
        D_low = 0.29 * Jz / nu_ratio
        D_mid = 0.35 * Jz * J**-0.5 / nu_ratio * kr**0.5
        D_high = 1.2 * Jz**-0.5 * J**-0.5 * nu_ratio**0.5 * kr**0.5
        if JRz_break < J:
            D_JzJz, dDz = (
                (D_high, -0.5 * D_high / Jz)
                if Jz > Jz_break
                else (
                    D_mid,
                    D_mid / Jz,
                )
            )
        else:
            D_JzJz, dDz = D_low, D_low / Jz
        A_Jz = 0.5 * dDz
        Dzz = D_JzJz / (2 * nu * Jz)
        Dvzvz = nu * D_JzJz / (2 * Jz)
        Cz = A_Jz / (2 * Jz) - D_JzJz / (4 * Jz**2)
        Az, Avz = z * Cz, vz * Cz

    # everything above is in (kpc, km/s, Gyr); convert to the target unit system
    X, V, T = kpc, kms, Gyr
    conv = {
        "AR": X / T,
        "Aphi": 1 / T,
        "Az": X / T,
        "AvR": V / T,
        "AvT": V / T,
        "Avz": V / T,
        "Drr": X**2 / T,
        "DrvT": X * V / T,
        "DvTvT": V**2 / T,
        "DvRvR": V**2 / T,
        "DphivR": V / T,
        "Dphph": 1 / T,
        "Dzz": X**2 / T,
        "Dvzvz": V**2 / T,
    }
    loc = locals()
    c = {k: loc[k] * (1 * conv[k]).decompose(units).value for k in conv}
    Rn = R * (1 * kpc).decompose(units).value
    mu = np.array([c["AR"], Rn * c["Aphi"], c["Az"], c["AvR"], c["AvT"], c["Avz"]])
    M = np.zeros((6, 6))
    for (i, k), val in {
        (0, 0): c["Drr"],
        (0, 4): c["DrvT"],
        (4, 4): c["DvTvT"],
        (3, 3): c["DvRvR"],
        (1, 3): Rn * c["DphivR"],
        (1, 1): Rn**2 * c["Dphph"],
        (2, 2): c["Dzz"],
        (5, 5): c["Dvzvz"],
    }.items():
        M[i, k] = M[k, i] = val
    return mu, M


@pytest.mark.parametrize(
    "opts",
    [
        {},
        {"planar_suppression": False, "vertical": False},
        {"environmental_scaling": True},
        {"long_wavelength_only": True},
    ],
)
def test_winter27_matches_paper_formulas(opts):
    """C coefficients match an independent transcription of App. B / the script."""
    from gala.dynamics.diffusion import ISMDiffusionWinter27

    pot = gp.MilkyWayPotential(version="v2", units=galactic)
    model = ISMDiffusionWinter27(pot, R_min=2 * u.kpc, R_max=30 * u.kpc, **opts)
    J_floor = (1e-3 * u.kpc * u.km / u.s).decompose(galactic).value
    rng = np.random.default_rng(42)
    # kicks spanning both regimes of each fit
    for R, kick in zip(rng.uniform(5, 12, 8), np.geomspace(0.002, 0.06, 8)):
        vc = pot.circular_velocity([R, 0, 0] * u.kpc).decompose(galactic).value[0]
        ph = rng.uniform(0, 2 * np.pi)
        xv = np.array(
            [
                R * np.cos(ph),
                R * np.sin(ph),
                0.1,
                kick * np.cos(ph) - vc * np.sin(ph),
                kick * np.sin(ph) + vc * np.cos(ph),
                0.5 * kick,
            ]
        )
        mu, D = model.coefficients(xv.reshape(6, 1))
        mu_ref, D_ref = _winter27_reference(pot, xv, galactic, J_floor, **opts)
        np.testing.assert_allclose(
            mu[0], mu_ref, rtol=1e-3, atol=1e-12 * np.abs(mu_ref).max()
        )
        np.testing.assert_allclose(D[0], D_ref, rtol=1e-3, atol=0)


def test_winter27_round_trip_to_actions():
    """Projecting the x-v (mu, D) onto epicyclic actions, averaged over the epicycle
    phases, recovers the action-space (A_J, D_J) the model was built from."""
    from gala.dynamics.diffusion import ISMDiffusionWinter27

    vc = (235 * u.km / u.s).decompose(galactic).value
    pot = gp.LogarithmicPotential(
        v_c=235 * u.km / u.s, r_h=1e-3 * u.kpc, q1=1, q2=1, q3=0.9, units=galactic
    )
    model = ISMDiffusionWinter27(pot, planar_suppression=False)
    Ju = (1 * u.kpc * u.km / u.s).decompose(galactic).value
    Du = Ju**2 / (1 * u.Gyr).decompose(galactic).value
    Rg = 8.0
    nu_Rg = np.sqrt(pot.hessian([Rg, 0, 0] * u.kpc).decompose(galactic).value[2, 2, 0])

    def actions(c):  # flat curve: R_g = L_z/v_c, kappa = sqrt(2) v_c/R_g, nu ~ 1/R_g
        R, _, z, vR, vp, vz = c
        Lz = R * vp
        rg = Lz / vc
        ka, nu = np.sqrt(2) * vc / rg, nu_Rg * Rg / rg
        return np.array(
            [
                (vR**2 + ka**2 * (R - rg) ** 2) / (2 * ka),
                Lz,
                (vz**2 + nu**2 * z**2) / (2 * nu),
            ]
        )

    jR = jz = 0.3  # in kpc km/s; long-wavelength in-plane, low-Jz vertical regime
    ka = np.sqrt(2) * vc / Rg
    aR, az = np.sqrt(2 * jR * Ju / ka), np.sqrt(2 * jz * Ju / nu_Rg)
    h = 1e-4 * np.array([1, 0, 0.1, vc, vc, vc])
    D_J, A_J, n = np.zeros(3), np.zeros(3), 0
    for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        for thz in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            R = Rg - aR * np.cos(th)
            c = np.array(
                [
                    R,
                    0.0,
                    az * np.cos(thz),
                    ka * aR * np.sin(th),
                    Rg * vc / R,
                    az * nu_Rg * np.sin(thz),
                ]
            )
            mu, D = (a[0] for a in model.coefficients(c.reshape(6, 1)))
            G, Hs = np.zeros((3, 6)), np.zeros((3, 6, 6))
            for i in np.flatnonzero(h):
                e = np.eye(6)[i] * h[i]
                G[:, i] = (actions(c + e) - actions(c - e)) / (2 * h[i])
                for j in np.flatnonzero(h):
                    f = np.eye(6)[j] * h[j]
                    Hs[:, i, j] = (
                        actions(c + e + f)
                        - actions(c + e - f)
                        - actions(c - e + f)
                        + actions(c - e - f)
                    ) / (4 * h[i] * h[j])
            D_J += np.diag(G @ D @ G.T)
            A_J += G @ mu + 0.5 * np.einsum("kij,ij->k", Hs, D)
            n += 1

    # the fits' kappa/nu rescaling evaluated at 8 kpc
    kr = (ka / (np.sqrt(2) * 28 * u.km / u.s / u.kpc).decompose(galactic)).value
    nr = (nu_Rg / (2 * np.pi / 75.55 / u.Myr).decompose(galactic)).value
    np.testing.assert_allclose(
        D_J / n / Du, [9.3 * jR / kr, 6900.0, 0.29 * jz / nr], rtol=0.01
    )
    np.testing.assert_allclose(
        A_J / n / (Du / Ju), [4.65 / kr, 0.0, 0.145 / nr], rtol=0.01, atol=1e-3
    )


def test_winter27_integrates():
    """Smoke test: finite orbits, and the J_phi spread follows sqrt(D_JphiJphi t)."""
    from gala.dynamics.diffusion import ISMDiffusionWinter27

    pot = gp.LogarithmicPotential(
        v_c=235 * u.km / u.s, r_h=1e-3 * u.kpc, q1=1, q2=1, q3=0.9, units=galactic
    )
    vc = (235 * u.km / u.s).decompose(galactic).value
    N = 2000
    rng = np.random.default_rng(1)
    R, ph = rng.uniform(7.5, 8.5, N), rng.uniform(0, 2 * np.pi, N)
    w0 = gd.PhaseSpacePosition(
        pos=np.array([R * np.cos(ph), R * np.sin(ph), np.zeros(N)]) * u.kpc,
        vel=np.array([-vc * np.sin(ph), vc * np.cos(ph), np.zeros(N)]) * u.kpc / u.Myr,
    )
    orbit = pot.integrate_orbit(
        w0,
        dt=0.5,
        n_steps=200,
        diffusion=ISMDiffusionWinter27(pot),
        kick_seed=3,
        save_all=False,
    )
    assert np.all(np.isfinite(orbit.w(galactic)))
    dLz = orbit.angular_momentum()[2, -1] - w0.angular_momentum()[2]
    expected = np.sqrt(6900 * 0.1)  # kpc km/s, D_JphiJphi * 100 Myr
    rms = np.std(dLz.to_value(u.kpc * u.km / u.s))
    assert 0.8 * expected < rms < 1.4 * expected
