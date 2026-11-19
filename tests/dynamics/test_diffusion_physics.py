"""Physics checks of the in-integrator diffusion kick: analytic SDE solutions,
convergence with the kick interval, and invariances.

The statistical tests use fixed seeds (so they are deterministic) and ensembles large
enough that each tolerance is at least ~5 standard deviations of the seed-to-seed
scatter, measured over several seeds.
"""

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

from gala.dynamics.diffusion import ConstantDiffusion

# A cored logarithmic potential is harmonic, with omega = v_c / r_h, well inside its
# core; the oscillator tests keep |x| < ~1e-2 r_h so anharmonic terms are < ~1e-4.
VC, RH = 0.1, 1.0  # kpc/Myr, kpc
OMEGA = VC / RH  # 1/Myr
D_OSC = 2e-10  # kpc^2/Myr^3, velocity diffusion along x


def _oscillator():
    return gp.LogarithmicPotential(
        v_c=VC * u.kpc / u.Myr, r_h=RH * u.kpc, q1=1, q2=1, q3=1, units=galactic
    )


def _oscillator_moments(t):
    """<x^2>, <v^2>, <x v> for x'' = -omega^2 x + sqrt(D) xi(t), from x = v = 0."""
    s = np.sin(2 * OMEGA * t) / (2 * OMEGA)
    return (
        D_OSC / (2 * OMEGA**2) * (t - s),
        D_OSC / 2 * (t + s),
        D_OSC * np.sin(OMEGA * t) ** 2 / (2 * OMEGA**2),
    )


def _run_oscillator(N, dt, n_steps, kick_every=1, seed=1):
    """Ensemble moments of (x, v_x) after n_steps, all orbits starting at rest."""
    D = np.zeros((6, 6))
    D[3, 3] = D_OSC
    diff = ConstantDiffusion(
        D=D, basis="cartesian", units=galactic, kick_every=kick_every
    )
    w0 = gd.PhaseSpacePosition(
        pos=np.zeros((3, N)) * u.kpc, vel=np.zeros((3, N)) * u.kpc / u.Myr
    )
    orbit = _oscillator().integrate_orbit(
        w0, dt=dt, n_steps=n_steps, save_all=False, diffusion=diff, kick_seed=seed
    )
    x = orbit.x[-1].to_value(u.kpc)
    v = orbit.v_x[-1].to_value(u.kpc / u.Myr)
    return np.mean(x**2), np.mean(v**2), np.mean(x * v)


def _disc(pot, N, seed=0):
    """Circular orbits in the midplane at uniform R in [4, 12] kpc."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(4, 12, N)
    phi = rng.uniform(0, 2 * np.pi, N)
    vc = pot.circular_velocity(R=R * u.kpc).to_value(u.kpc / u.Myr)
    return gd.PhaseSpacePosition(
        pos=[R * np.cos(phi), R * np.sin(phi), np.zeros(N)] * u.kpc,
        vel=[-vc * np.sin(phi), vc * np.cos(phi), np.zeros(N)] * u.kpc / u.Myr,
    )


def _isotropic_velocity_diffusion(Dv, basis, units=galactic):
    D = np.zeros((6, 6))
    D[3, 3] = D[4, 4] = D[5, 5] = Dv
    return ConstantDiffusion(D=D, basis=basis, units=units)


def _energy(pot, w):
    v2 = np.sum(w.v_xyz.to_value(u.kpc / u.Myr) ** 2, axis=0)
    return pot.energy(w.xyz).to_value(u.kpc**2 / u.Myr**2) + 0.5 * v2


# ---- analytic SDE solutions ----


def test_stochastic_harmonic_oscillator():
    """Ensemble second moments match the analytic stochastic-oscillator solution.

    With omega*dt = 0.025 the scheme's first-order bias is ~0.5%; the tolerances
    are ~5x the seed-to-seed scatter at N = 5e4.
    """
    N, dt, n_steps = 50_000, 0.25, 188  # omega * T = 4.7
    x2, v2, xv = _run_oscillator(N, dt, n_steps)
    x2_true, v2_true, xv_true = _oscillator_moments(dt * n_steps)
    np.testing.assert_allclose(x2, x2_true, rtol=0.05)
    np.testing.assert_allclose(v2, v2_true, rtol=0.03)
    # <x v> is small relative to its scatter, so compare the correlation coefficient
    rho, rho_true = xv / np.sqrt(x2 * v2), xv_true / np.sqrt(x2_true * v2_true)
    assert abs(rho - rho_true) < 5 / np.sqrt(N)


@pytest.mark.parametrize("basis", ["cartesian", "cylindrical"])
def test_energy_heating_rate(basis):
    """Isotropic velocity diffusion D heats any potential at d<E>/dt = (3/2) D (Ito).

    The cylindrical basis applies the kick in curvilinear coordinates; with velocity
    noise only, it must heat at the same rate.
    """
    pot = gp.MilkyWayPotential(version="v2", units=galactic)
    Dv, dt, n_steps, N = 1.25e-5, 0.5, 400, 50_000  # sigma_v -> ~50 km/s
    w0 = _disc(pot, N)
    orbit = pot.integrate_orbit(
        w0,
        dt=dt,
        n_steps=n_steps,
        save_all=False,
        diffusion=_isotropic_velocity_diffusion(Dv, basis),
        kick_seed=2,
    )
    dE = _energy(pot, orbit[-1]) - _energy(pot, w0)
    expected = 1.5 * Dv * dt * n_steps
    # seed-to-seed scatter of the mean is ~2% at this N
    np.testing.assert_allclose(dE.mean(), expected, rtol=0.1)


def test_position_drift_is_exact():
    """A constant velocity drift moves a free particle by mu dt^2 N(N-1)/2: the drift
    is applied at the synchronized state at the end of each step."""
    mu_v = np.array([0.01, -0.02, 0.005])
    drift = np.concatenate([np.zeros(3), mu_v])
    diff = ConstantDiffusion(
        D=np.zeros((6, 6)), drift=drift, basis="cartesian", units=galactic
    )
    H = gp.Hamiltonian(gp.NullPotential(units=galactic))
    w0 = gd.PhaseSpacePosition(pos=[0, 0, 0.0] * u.kpc, vel=[0, 0, 0.0] * u.kpc / u.Myr)
    dt, n = 0.5, 80
    orbit = H.integrate_orbit(w0, dt=dt, n_steps=n, diffusion=diff, kick_seed=1)
    np.testing.assert_allclose(
        orbit.xyz[:, -1].to_value(u.kpc), mu_v * dt**2 * n * (n - 1) / 2, rtol=1e-12
    )


# ---- convergence with the kick interval ----


def test_weak_convergence_is_first_order():
    """The bias of the oscillator moments shrinks linearly with the kick interval h.

    Measured bias is ~0.21 omega h in both <x^2> (low) and <v^2> (high), with a
    seed-to-seed scatter of < ~0.5% at N = 1e5, well below the bias at these h.
    """
    N, dt, n_steps = 100_000, 0.5, 94  # omega * T = 4.7
    x2_true, v2_true, _ = _oscillator_moments(dt * n_steps)
    kick_every = np.array([2, 4, 8, 16])
    bias = []
    for k in kick_every:
        x2, v2, _ = _run_oscillator(N, dt, n_steps, kick_every=int(k))
        bias.append(0.5 * (abs(x2 / x2_true - 1) + abs(v2 / v2_true - 1)))
    bias = np.array(bias)
    assert np.all(np.diff(bias) > 0)
    slope = np.polyfit(np.log(kick_every), np.log(bias), 1)[0]
    assert 0.7 < slope < 1.3


# ---- invariances ----


def test_unit_system_invariance():
    """The same physical setup in kpc and in pc gives the same orbits (same noise)."""
    Dv, dt, n_steps = 1e-5, 0.5, 400  # kpc^2 / Myr^3
    units_pc = UnitSystem(u.pc, u.Myr, u.Msun, u.radian)
    pot_kpc = gp.MilkyWayPotential(version="v2", units=galactic)
    pot_pc = gp.MilkyWayPotential(version="v2", units=units_pc)
    w0 = _disc(pot_kpc, 100)

    o_kpc = pot_kpc.integrate_orbit(
        w0,
        dt=dt,
        n_steps=n_steps,
        save_all=False,
        diffusion=_isotropic_velocity_diffusion(Dv, "cylindrical"),
        kick_seed=7,
    )
    o_pc = pot_pc.integrate_orbit(
        w0,
        dt=dt,
        n_steps=n_steps,
        save_all=False,
        diffusion=_isotropic_velocity_diffusion(Dv * 1e6, "cylindrical", units_pc),
        kick_seed=7,
    )
    x_kpc = o_kpc.xyz[:, -1].to_value(u.kpc)
    np.testing.assert_allclose(
        o_pc.xyz[:, -1].to_value(u.kpc), x_kpc, atol=1e-9 * np.abs(x_kpc).max()
    )


def test_save_all_invariance():
    """save_all only changes what is returned, not the stochastic trajectory."""
    pot = gp.MilkyWayPotential(version="v2", units=galactic)
    w0 = _disc(pot, 50)
    kw = {
        "dt": 0.5,
        "n_steps": 200,
        "diffusion": _isotropic_velocity_diffusion(1e-5, "cylindrical"),
        "kick_seed": 3,
    }
    o_all = pot.integrate_orbit(w0, save_all=True, **kw)
    o_end = pot.integrate_orbit(w0, save_all=False, **kw)
    np.testing.assert_array_equal(o_all.xyz[:, -1].value, o_end.xyz[:, -1].value)
    np.testing.assert_array_equal(o_all.v_xyz[:, -1].value, o_end.v_xyz[:, -1].value)


def test_basis_invariance_of_isotropic_velocity_covariance():
    """Isotropic velocity diffusion has the same Cartesian velocity covariance growth
    whether it is specified in the cylindrical or the Cartesian basis."""
    N, Dv, dt, n_steps = 40_000, 1e-4, 1.0, 20
    rng = np.random.default_rng(4)
    pos = rng.uniform(-10, 10, (3, N))
    H = gp.Hamiltonian(gp.NullPotential(units=galactic))
    w0 = gd.PhaseSpacePosition(pos=pos * u.kpc, vel=np.zeros((3, N)) * u.kpc / u.Myr)
    T = dt * n_steps
    for basis in ["cartesian", "cylindrical"]:
        orbit = H.integrate_orbit(
            w0,
            dt=dt,
            n_steps=n_steps,
            save_all=False,
            diffusion=_isotropic_velocity_diffusion(Dv, basis),
            kick_seed=5,
        )
        cov = np.cov(orbit.v_xyz[:, -1].to_value(u.kpc / u.Myr))
        # diagonal: relative scatter sqrt(2/N) ~ 0.7%; off-diagonal: ~ Dv T / sqrt(N)
        np.testing.assert_allclose(np.diag(cov), Dv * T, rtol=0.035)
        off = cov[~np.eye(3, dtype=bool)]
        assert np.all(np.abs(off) < 5 * Dv * T / np.sqrt(N))


@pytest.mark.parametrize("kind", ["builtin", "composite", "time_interpolated"])
def test_zero_diffusion_matches_leapfrog(kind):
    """D = 0 reproduces plain leapfrog exactly for each C potential path."""
    if kind == "builtin":
        pot = gp.HernquistPotential(m=1e11, c=10, units=galactic)
    elif kind == "composite":
        pot = gp.MilkyWayPotential(version="v2", units=galactic)
    else:
        pot = gp.TimeInterpolatedPotential(
            gp.HernquistPotential,
            np.array([0.0, 250.0, 500.0]) * u.Myr,
            m=np.array([1e11, 1.1e11, 1.2e11]) * u.Msun,
            c=10 * u.kpc,
            units=galactic,
        )
    w0 = gd.PhaseSpacePosition(
        pos=[10.0, 0, 1] * u.kpc, vel=[0, 175.0, 10] * u.km / u.s
    )
    diff0 = ConstantDiffusion(D=np.zeros((6, 6)), basis="cylindrical", units=galactic)
    o_diff = pot.integrate_orbit(w0, dt=1.0, n_steps=500, diffusion=diff0, kick_seed=1)
    o_plain = pot.integrate_orbit(w0, dt=1.0, n_steps=500)
    np.testing.assert_allclose(o_diff.xyz.value, o_plain.xyz.value, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        o_diff.v_xyz.value, o_plain.v_xyz.value, rtol=0, atol=1e-12
    )
