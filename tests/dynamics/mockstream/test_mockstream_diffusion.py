"""Diffusion kicks during mock-stream generation (leapfrog only)."""

import astropy.units as u
import numpy as np
import pytest
from gala._cconfig import GSL_ENABLED

from gala.dynamics import FardalStreamDF, PhaseSpacePosition
from gala.dynamics.mockstream.mockstream_generator import MockStreamGenerator
from gala.dynamics.nbody import DirectNBody
from gala.potential import Hamiltonian, HernquistPotential, NFWPotential
from gala.units import UnitSystem, galactic

if not GSL_ENABLED:
    pytest.skip("diffusion depends on GSL", allow_module_level=True)

from gala.dynamics.diffusion import ConstantDiffusion

POT = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.0, units=galactic)
PROG_W0 = PhaseSpacePosition(
    pos=[15.0, 0.0, 0] * u.kpc, vel=[0, 0, 0.13] * u.kpc / u.Myr
)
PROG_MASS = 2.5e4 * u.Msun


def _vel_diffusion(D, **kwargs):
    return ConstantDiffusion(
        np.diag([0, 0, 0, D, D, D]), basis="cartesian", units=galactic, **kwargs
    )


def _run(diffusion=None, kick_seed=None, nbody=None, n_particles=4, **kwargs):
    # a fresh DF random state each time, so every run releases the same particles
    df = FardalStreamDF(gala_modified=True, random_state=np.random.default_rng(1))
    gen = MockStreamGenerator(df=df, hamiltonian=Hamiltonian(POT))
    kwargs.setdefault("Integrator", "leapfrog")
    return gen.run(
        PROG_W0,
        PROG_MASS,
        dt=1.0,
        n_steps=200,
        n_particles=n_particles,
        nbody=nbody,
        diffusion=diffusion,
        kick_seed=kick_seed,
        **kwargs,
    )


def test_zero_diffusion_matches_no_diffusion():
    stream0, prog0 = _run()
    stream, prog = _run(_vel_diffusion(0.0), kick_seed=1)
    np.testing.assert_array_equal(stream.xyz.value, stream0.xyz.value)
    np.testing.assert_array_equal(stream.v_xyz.value, stream0.v_xyz.value)
    np.testing.assert_array_equal(prog.xyz.value, prog0.xyz.value)


def test_seeded_runs_reproduce():
    a, _ = _run(_vel_diffusion(1e-6), kick_seed=3)
    b, _ = _run(_vel_diffusion(1e-6), kick_seed=3)
    c, _ = _run(_vel_diffusion(1e-6), kick_seed=4)
    np.testing.assert_array_equal(a.xyz.value, b.xyz.value)
    assert not np.array_equal(a.xyz.value, c.xyz.value)


def test_stream_heating_rate():
    """Isotropic velocity diffusion raises each released particle's energy by
    (3/2) D (t_final - t_release) on average (Ito), in any static potential."""
    D = 5e-5  # (kpc/Myr)^2 / Myr
    stream0, _ = _run(n_particles=16)
    stream, _ = _run(_vel_diffusion(D), kick_seed=5, n_particles=16)
    dE = (stream.energy(POT) - stream0.energy(POT)).decompose(galactic).value
    tau = 200.0 - stream.release_time.decompose(galactic).value
    np.testing.assert_allclose(dE.sum() / tau.sum(), 1.5 * D, rtol=0.15)


def test_drift_matches_integrate_orbit():
    """A deterministic (drift-only) model with position kicks and kick_every > 1 moves
    each released batch exactly as integrate_orbit does from its release state."""
    drift = np.array([1e-3, 0, -5e-4, 0, 1e-5, 0])
    diff = ConstantDiffusion(
        np.zeros((6, 6)), drift=drift, basis="cartesian", units=galactic, kick_every=3
    )
    stream0, _ = _run()
    stream, _ = _run(diff)
    H = Hamiltonian(POT)
    t_rel = stream0.release_time.decompose(galactic).value
    for t1 in np.unique(t_rel)[[0, 50, 151]]:
        idx = t_rel == t1
        w_end = PhaseSpacePosition(pos=stream0.xyz[:, idx], vel=stream0.v_xyz[:, idx])
        # recover the release state by reversing plain leapfrog
        back = H.integrate_orbit(
            w_end, t=np.arange(200.0, t1 - 0.5, -1.0), Integrator="leapfrog"
        )[-1]
        fwd = H.integrate_orbit(back, t=np.arange(t1, 200.5, 1.0), diffusion=diff)[-1]
        np.testing.assert_allclose(
            fwd.xyz.value, stream.xyz.value[:, idx], rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(
            fwd.v_xyz.value, stream.v_xyz.value[:, idx], rtol=1e-8, atol=1e-12
        )


def test_nbody_bodies_are_not_kicked():
    """Only the stream particles diffuse: the progenitor and perturbers don't."""
    nbody = DirectNBody(
        w0=PhaseSpacePosition([20, 0, 0] * u.kpc, [0, 100, 0] * u.km / u.s),
        external_potential=POT,
        particle_potentials=[
            HernquistPotential(m=1e8 * u.Msun, c=0.2 * u.kpc, units=galactic)
        ],
    )
    _, bodies0 = _run(nbody=nbody)
    _, bodies = _run(_vel_diffusion(1e-5), kick_seed=1, nbody=nbody)
    np.testing.assert_array_equal(bodies.xyz.value, bodies0.xyz.value)


def test_unsupported_options_raise(tmp_path):
    with pytest.raises(NotImplementedError, match="leapfrog"):
        _run(_vel_diffusion(1e-6), Integrator="dop853")
    with pytest.raises(NotImplementedError, match="output_every"):
        _run(
            _vel_diffusion(1e-6),
            output_every=10,
            output_filename=str(tmp_path / "s.hdf5"),
        )
    pc_units = UnitSystem(u.pc, u.Myr, u.Msun, u.radian)
    diff = ConstantDiffusion(np.zeros((6, 6)), basis="cartesian", units=pc_units)
    with pytest.raises(ValueError, match="units"):
        _run(diff)
