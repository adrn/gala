"""Tests for the trial SDE / diffusion integrator (gala.dynamics.diffusion)."""

import astropy.units as u
import numpy as np
import pytest
from gala._cconfig import GSL_ENABLED

import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

if not GSL_ENABLED:
    pytest.skip(
        "skipping diffusion integrator tests: they depend on GSL",
        allow_module_level=True,
    )

from gala.dynamics.diffusion import (
    ConstantDiffusion,
    ConstantTensorDiffusion,
    ExampleRadialDiffusion,
    StochasticOrbitIntegrator,
)


def test_diffusion_off_matches_leapfrog():
    """With zero diffusion the integrator must reproduce plain leapfrog exactly."""
    pot = gp.HernquistPotential(m=1e11, c=10, units=galactic)
    diff = ConstantDiffusion(D=[0.0, 0.0, 0.0], units=galactic)
    soi = StochasticOrbitIntegrator(pot, diff, seed=42)

    w0 = gd.PhaseSpacePosition(pos=[10.0, 0, 0] * u.kpc, vel=[0, 150.0, 0] * u.km / u.s)

    orbit = soi.integrate_orbit(w0, dt=1.0, n_steps=500)

    ref = gp.Hamiltonian(pot).integrate_orbit(
        w0, dt=1.0, n_steps=500, Integrator="leapfrog"
    )

    assert np.allclose(orbit.xyz.to_value(u.kpc), ref.xyz.to_value(u.kpc), atol=1e-12)
    assert np.allclose(
        orbit.v_xyz.to_value(u.kpc / u.Myr),
        ref.v_xyz.to_value(u.kpc / u.Myr),
        atol=1e-12,
    )


def test_reproducibility():
    """Same seed -> identical orbits; different seed -> different orbits."""
    pot = gp.NullPotential(units=galactic)
    diff = ConstantDiffusion(D=[1e-4, 1e-4, 1e-4], units=galactic)
    w0 = gd.PhaseSpacePosition(pos=[0, 0, 0] * u.kpc, vel=[0, 0, 0] * u.kpc / u.Myr)

    o_a = StochasticOrbitIntegrator(pot, diff, seed=555).integrate_orbit(
        w0, dt=1.0, n_steps=100
    )
    o_b = StochasticOrbitIntegrator(pot, diff, seed=555).integrate_orbit(
        w0, dt=1.0, n_steps=100
    )
    o_c = StochasticOrbitIntegrator(pot, diff, seed=556).integrate_orbit(
        w0, dt=1.0, n_steps=100
    )

    assert np.array_equal(o_a.xyz.value, o_b.xyz.value)
    assert np.array_equal(o_a.v_xyz.value, o_b.v_xyz.value)
    assert not np.array_equal(o_a.v_xyz.value, o_c.v_xyz.value)


def test_velocity_variance_growth():
    """Free particle + constant diagonal D: Var(v_i) should grow as D_i * T."""
    N = 4000
    D = np.array([1e-4, 4e-4, 2.5e-4])  # (kpc/Myr)^2 / Myr
    pot = gp.NullPotential(units=galactic)
    diff = ConstantDiffusion(D=D, units=galactic)
    soi = StochasticOrbitIntegrator(pot, diff, seed=123)

    w0 = gd.PhaseSpacePosition(
        pos=np.zeros((3, N)) * u.kpc, vel=np.zeros((3, N)) * u.kpc / u.Myr
    )
    dt, n_steps = 0.5, 200
    T = dt * n_steps

    orbit = soi.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    var = orbit[-1].v_xyz.to_value(u.kpc / u.Myr).var(axis=1)

    # ~1/sqrt(N) statistical tolerance
    assert np.allclose(var, D * T, rtol=0.1)


def test_full_tensor_covariance():
    """Free particle + full tensor D: ensemble velocity covariance ~ D * T."""
    N = 6000
    D = np.array(
        [
            [3e-4, 1e-4, 0.0],
            [1e-4, 2e-4, 0.0],
            [0.0, 0.0, 1e-4],
        ]
    )
    pot = gp.NullPotential(units=galactic)
    diff = ConstantTensorDiffusion(D=D, units=galactic)
    soi = StochasticOrbitIntegrator(pot, diff, seed=7)

    w0 = gd.PhaseSpacePosition(
        pos=np.zeros((3, N)) * u.kpc, vel=np.zeros((3, N)) * u.kpc / u.Myr
    )
    dt, n_steps = 0.5, 200
    T = dt * n_steps

    orbit = soi.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    cov = np.cov(orbit[-1].v_xyz.to_value(u.kpc / u.Myr))

    target = D * T
    mask = target != 0
    assert np.max(np.abs((cov - target)[mask] / target[mask])) < 0.1
    # near-zero entries stay small
    assert np.all(np.abs(cov[~mask]) < 0.1 * target[mask].min())


def test_shapes_and_save_all():
    """Integrating N orbits at once and save_all=False return sane shapes."""
    N = 5
    pot = gp.HernquistPotential(m=1e11, c=10, units=galactic)
    diff = ConstantDiffusion(D=[1e-5, 1e-5, 1e-5], units=galactic)
    soi = StochasticOrbitIntegrator(pot, diff, seed=99)

    pos = np.zeros((3, N)) * u.kpc
    pos[0] = np.linspace(8, 12, N) * u.kpc
    vel = np.zeros((3, N)) * u.km / u.s
    vel[1] = np.linspace(150, 200, N) * u.km / u.s
    w0 = gd.PhaseSpacePosition(pos=pos, vel=vel)

    orbit = soi.integrate_orbit(w0, dt=1.0, n_steps=100)
    assert orbit.shape == (101, N)

    orbit_final = soi.integrate_orbit(w0, dt=1.0, n_steps=100, save_all=False)
    assert orbit_final.shape == (1, N)


def test_position_dependent_diffusion_runs():
    """The template position-dependent model integrates and heats the orbit."""
    N = 2000
    pot = gp.HernquistPotential(m=1e11, c=10, units=galactic)
    diff = ExampleRadialDiffusion(
        D=[1e-4, 1e-4, 1e-4], r_s=10.0 * u.kpc, units=galactic
    )
    soi = StochasticOrbitIntegrator(pot, diff, seed=321)

    pos = np.tile([[8.0], [0.0], [0.0]], (1, N)) * u.kpc
    vel = np.tile([[0.0], [180.0], [0.0]], (1, N)) * u.km / u.s
    w0 = gd.PhaseSpacePosition(pos=pos, vel=vel)

    orbit = soi.integrate_orbit(w0, dt=1.0, n_steps=500)

    # velocity dispersion should grow relative to the (identical) initial state
    v0 = orbit[0].v_xyz.to_value(u.kpc / u.Myr).std(axis=1)
    vf = orbit[-1].v_xyz.to_value(u.kpc / u.Myr).std(axis=1)
    assert np.all(v0 < 1e-12)  # started identical
    assert np.all(vf > 0)


def test_requires_diffusion_base():
    """Passing a non-DiffusionBase object raises a clear error."""
    pot = gp.NullPotential(units=galactic)
    with pytest.raises(TypeError):
        StochasticOrbitIntegrator(pot, diffusion="not a model")


def test_bad_parameter_shape():
    """A wrongly shaped diffusion parameter raises a clear error."""
    with pytest.raises(ValueError):
        ConstantDiffusion(D=[1e-4, 1e-4], units=galactic)  # needs length 3
