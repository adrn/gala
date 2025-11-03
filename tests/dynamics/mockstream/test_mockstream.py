import itertools
import os
import time

import astropy.units as u
import numpy as np
import pytest

import gala.integrate as gi
from gala._optional_deps import HAS_H5PY
from gala.dynamics import ChenStreamDF, FardalStreamDF, Orbit, PhaseSpacePosition
from gala.dynamics.mockstream import MockStream
from gala.dynamics.mockstream.mockstream_generator import MockStreamGenerator
from gala.dynamics.nbody import DirectNBody
from gala.potential import (
    ConstantRotatingFrame,
    Hamiltonian,
    HernquistPotential,
    NFWPotential,
)
from gala.units import galactic


@pytest.fixture
def rng():
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(1234)


@pytest.fixture
def basic_potential():
    """Standard NFW potential for most tests."""
    return NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.0, units=galactic)


@pytest.fixture
def basic_hamiltonian(basic_potential):
    """Standard Hamiltonian with NFW potential."""
    return Hamiltonian(basic_potential)


@pytest.fixture
def progenitor_w0():
    """Standard progenitor initial conditions."""
    return PhaseSpacePosition(
        pos=[15.0, 0.0, 0] * u.kpc, vel=[0, 0, 0.13] * u.kpc / u.Myr
    )


@pytest.fixture
def progenitor_mass():
    """Standard progenitor mass."""
    return 2.5e4 * u.Msun


@pytest.fixture
def basic_stream_generator(basic_hamiltonian, rng):
    """Basic MockStreamGenerator with Fardal DF."""
    df = FardalStreamDF(gala_modified=True, random_state=rng)
    return MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)


# Tests


def test_init(rng, basic_hamiltonian, basic_potential, progenitor_w0):
    """Test MockStreamGenerator initialization and validation."""
    df = FardalStreamDF(gala_modified=True, random_state=rng)

    # Test that invalid arguments are caught
    with pytest.raises(TypeError):
        MockStreamGenerator(df="some df", hamiltonian=basic_hamiltonian)

    with pytest.raises(TypeError):
        MockStreamGenerator(
            df=df, hamiltonian=basic_hamiltonian, progenitor_potential="stuff"
        )

    # Test validating the input nbody
    nbody_w0 = PhaseSpacePosition(
        pos=[25.0, 0.0, 0] * u.kpc, vel=[0, 0, 0.13] * u.kpc / u.Myr
    )

    # Different external potential should fail
    potential2 = NFWPotential.from_circular_velocity(v_c=0.2, r_s=25.0, units=galactic)
    nbody = DirectNBody(
        w0=nbody_w0, external_potential=potential2, particle_potentials=[None]
    )
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)
    with pytest.raises(ValueError):
        gen._get_nbody(progenitor_w0, nbody)

    # Different frame should fail
    frame2 = ConstantRotatingFrame([0, 0, 25.0] * u.km / u.s / u.kpc, units=galactic)
    nbody = DirectNBody(
        w0=nbody_w0,
        external_potential=basic_potential,
        frame=frame2,
        particle_potentials=[None],
    )
    with pytest.raises(ValueError):
        gen._get_nbody(progenitor_w0, nbody)

    # Should succeed with matching potential and frame
    nbody = DirectNBody(
        w0=nbody_w0, external_potential=basic_potential, particle_potentials=[None]
    )
    new_nbody = gen._get_nbody(progenitor_w0, nbody)


def test_run(rng, basic_hamiltonian, progenitor_w0, progenitor_mass):
    """Test basic stream generation functionality."""
    df = FardalStreamDF(gala_modified=True, random_state=rng)
    prog_pot = HernquistPotential(progenitor_mass, 4 * u.pc, units=galactic)

    # Basic run without self-gravity
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)
    stream1, _ = gen.run(progenitor_w0, progenitor_mass, dt=-1.0, n_steps=100)

    # Test that mass must have units
    with pytest.raises(TypeError):
        gen.run(progenitor_w0, progenitor_mass.value, dt=-1.0, n_steps=100)

    # With self-gravity - should produce different results
    gen = MockStreamGenerator(
        df=df, hamiltonian=basic_hamiltonian, progenitor_potential=prog_pot
    )
    stream2, _ = gen.run(progenitor_w0, progenitor_mass, dt=-1.0, n_steps=100)
    assert not u.allclose(stream1.xyz, stream2.xyz)

    # Test skipping release steps
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)
    stream3, _ = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=-1.0,
        n_steps=100,
        release_every=4,
        n_particles=4,
    )
    assert stream3.shape == ((100 // 4 + 1) * 4 * 2,)

    # Test custom n_particles array
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)
    n_particles = np.random.randint(0, 4, size=101)
    stream3, _ = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=-1.0,
        n_steps=100,
        release_every=1,
        n_particles=n_particles,
    )
    assert stream3.shape[0] == 2 * n_particles.sum()


@pytest.mark.parametrize("dt", [1, -1])
@pytest.mark.parametrize("save_all", [True, False])
@pytest.mark.parametrize("Integrator", [gi.LeapfrogIntegrator, gi.DOPRI853Integrator])
def test_mockstream_nbody_run(
    rng,
    basic_potential,
    basic_hamiltonian,
    progenitor_w0,
    progenitor_mass,
    dt,
    save_all,
    Integrator,
):
    """Test stream generation with N-body perturbers using different integrators."""
    df = FardalStreamDF(gala_modified=True, random_state=rng)

    # Test passing custom N-body with perturber
    nbody_w0 = PhaseSpacePosition([20, 0, 0] * u.kpc, [0, 100, 0] * u.km / u.s)
    nbody = DirectNBody(
        w0=nbody_w0,
        external_potential=basic_potential,
        particle_potentials=[
            NFWPotential(m=1e8 * u.Msun, r_s=0.2 * u.kpc, units=galactic)
        ],
        save_all=save_all,
    )
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)
    stream, prog = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=dt,
        n_steps=100,
        nbody=nbody,
        Integrator=Integrator,
    )

    # Basic sanity checks
    assert stream.shape[0] > 0
    assert np.isfinite(stream.xyz).all()
    assert np.isfinite(stream.v_xyz).all()


@pytest.mark.skipif(not HAS_H5PY, reason="h5py required for this test")
def test_nbody_hdf5_broadcast_bug(
    tmpdir, rng, basic_potential, basic_hamiltonian, progenitor_w0, progenitor_mass
):
    """
    Regression test for HDF5 broadcast bug when using nbody with output_filename,
    reported in #158.
    """
    import h5py

    prog_pot = HernquistPotential(progenitor_mass, 4 * u.pc, units=galactic)

    nbody_w0 = PhaseSpacePosition([20, 0, 0] * u.kpc, [0, 100, 0] * u.km / u.s)
    nbody = DirectNBody(
        w0=nbody_w0,
        external_potential=basic_potential,
        particle_potentials=[
            NFWPotential(m=1e8 * u.Msun, r_s=0.2 * u.kpc, units=galactic)
        ],
    )

    # Use trail=False and n_particles=1 to ensure we have fewer stream particles
    # than nbodies at early timesteps
    df = FardalStreamDF(gala_modified=True, trail=False, random_state=rng)
    gen = MockStreamGenerator(
        df=df, hamiltonian=basic_hamiltonian, progenitor_potential=prog_pot
    )

    filename = os.path.join(str(tmpdir), "test_nbody.hdf5")

    # This should trigger the bug if not fixed: at first output, n=1 but nbodies=2
    stream, prog = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=-1.0,
        n_steps=8,
        nbody=nbody,
        release_every=1,
        n_particles=1,
        output_every=1,
        output_filename=filename,
        check_filesize=False,
        overwrite=True,
    )

    # If we get here without error, verify the file was created correctly
    with h5py.File(filename, mode="r") as f:
        stream_orbits = Orbit.from_hdf5(f["stream"])
        nbody_orbits = Orbit.from_hdf5(f["nbody"])

        # Check that nbody has the correct shape
        assert nbody_orbits.shape[0] == 9  # noutput_times
        assert nbody_orbits.shape[1] == 2  # progenitor + 1 perturber

        # Check that values are finite
        assert np.isfinite(nbody_orbits.xyz).all()
        assert np.isfinite(nbody_orbits.v_xyz).all()


# TODO: add LeapfrogIntegrator if animation support added
@pytest.mark.parametrize(
    ("dt", "nsteps", "output_every", "release_every", "n_particles", "trail"),
    list(itertools.product([1, -1], [16, 17], [1, 2], [1, 4], [1, 4], [True, False])),
)
@pytest.mark.parametrize("Integrator", [gi.DOPRI853Integrator])
@pytest.mark.skipif(not HAS_H5PY, reason="h5py required for this test")
def test_animate(
    tmpdir,
    rng,
    basic_hamiltonian,
    progenitor_w0,
    progenitor_mass,
    dt,
    nsteps,
    output_every,
    release_every,
    n_particles,
    trail,
    Integrator,
):
    """Test animation output to HDF5 with various parameter combinations."""
    import h5py

    # The basic run with animation output
    df = FardalStreamDF(gala_modified=True, trail=trail, random_state=rng)
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)

    filename = os.path.join(str(tmpdir), f"test_{Integrator.__name__}.hdf5")
    _stream, _ = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=dt,
        n_steps=nsteps,
        release_every=release_every,
        n_particles=n_particles,
        output_every=output_every,
        output_filename=filename,
        overwrite=True,
        Integrator=Integrator,
    )

    with h5py.File(filename, mode="r") as f:
        stream_orbits = Orbit.from_hdf5(f["stream"])
        nbody_orbits = Orbit.from_hdf5(f["nbody"])

    noutput_times = 1 + nsteps // output_every
    if nsteps % output_every != 0:
        noutput_times += 1

    tail_n_particles = (1 + int(trail)) * n_particles
    expected_shape = (noutput_times, tail_n_particles * (nsteps // release_every + 1))

    assert stream_orbits.shape == expected_shape
    assert np.isfinite(stream_orbits[:, 0].xyz).all()
    assert np.isfinite(stream_orbits[:, 0].v_xyz).all()

    assert u.allclose(nbody_orbits.t, stream_orbits.t)

    assert np.isfinite(nbody_orbits.xyz).all()
    assert np.isfinite(nbody_orbits.v_xyz).all()
    assert np.isfinite(nbody_orbits.t).all()


@pytest.mark.xfail(reason="Timing comparison depends on system load...")
def test_integrator_kwargs_dop853(
    rng, basic_hamiltonian, progenitor_w0, progenitor_mass
):
    """Test that integrator kwargs are properly passed through."""
    df = ChenStreamDF(random_state=rng)
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)

    ti = time.time()
    stream1, _ = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=-1.0,
        n_steps=1000,
        Integrator_kwargs={"atol": 1e-12, "nmax": 0},
    )
    runtime1 = time.time() - ti

    ti = time.time()
    stream2, _ = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=-1.0,
        n_steps=1000,
        Integrator_kwargs={"atol": 1e-5, "nmax": 100, "err_if_fail": 0},
    )
    runtime2 = time.time() - ti

    print(f"stream 1, atol=1e-12, runtime = {runtime1}")
    print(f"stream 2, atol=1e-5, runtime = {runtime2}")

    assert runtime2 < runtime1


# ==============================================================================
# Integrator comparison tests
# ==============================================================================


def test_integrator_consistency_basic(
    rng, basic_hamiltonian, progenitor_w0, progenitor_mass
):
    """
    Test that Leapfrog and DOPRI853 produce qualitatively similar streams.

    Note: We don't expect exact agreement since they use different integration
    schemes (symplectic vs adaptive Runge-Kutta), but they should produce
    streams with similar overall structure.
    """
    df = FardalStreamDF(gala_modified=True, random_state=np.random.default_rng(123))
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)

    # Use same parameters for both integrators
    stream_dop, prog_dop = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=-1.0,
        n_steps=50,
        release_every=5,
        n_particles=2,
        Integrator=gi.DOPRI853Integrator,
    )

    # Run with Leapfrog - reinitialize generator to reset random state
    df = FardalStreamDF(gala_modified=True, random_state=np.random.default_rng(123))
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)

    stream_lf, prog_lf = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=-1.0,
        n_steps=50,
        release_every=5,
        n_particles=2,
        Integrator=gi.LeapfrogIntegrator,
    )

    # Check that both produced the same number of particles
    assert stream_dop.shape == stream_lf.shape

    # Check that progenitor final positions are reasonably close
    # (within ~1 kpc after 50 Myr integration)
    assert np.allclose(prog_dop.xyz.value, prog_lf.xyz.value, atol=1.0)

    # Check that stream positions are reasonably similar
    # We use a loose tolerance since integrators have different error profiles
    mean_sep = np.mean(
        np.linalg.norm(stream_dop.xyz.value - stream_lf.xyz.value, axis=0)
    )
    print(f"Mean particle separation between integrators: {mean_sep:.3f} kpc")
    assert mean_sep < 2.0  # Less than 2 kpc mean separation


@pytest.mark.parametrize("dt", [1.0, -1.0])
def test_integrator_energy_conservation(
    rng, basic_hamiltonian, progenitor_w0, progenitor_mass, dt
):
    """
    Test energy conservation for Leapfrog vs DOPRI853.

    Leapfrog is symplectic and should conserve energy better for long integrations.
    """
    df = FardalStreamDF(gala_modified=True, random_state=rng, trail=False)
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)

    # DOPRI853 with default tolerance
    _, prog_dop = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=dt,
        n_steps=200,
        release_every=200,  # Only release at beginning
        n_particles=1,
        Integrator="dop853",
    )
    E_dop_initial = basic_hamiltonian(prog_dop[0])
    E_dop_final = basic_hamiltonian(prog_dop[-1])
    dE_dop = float(np.abs((E_dop_final - E_dop_initial) / E_dop_initial))

    # Leapfrog
    _, prog_lf = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=dt,
        n_steps=200,
        release_every=200,  # Only release at beginning
        n_particles=1,
        Integrator="leapfrog",
    )
    E_lf_initial = basic_hamiltonian(prog_lf[0])
    E_lf_final = basic_hamiltonian(prog_lf[-1])
    dE_lf = float(np.abs((E_lf_final - E_lf_initial) / E_lf_initial))

    print(f"DOPRI853 relative energy error: {dE_dop:.6e}")
    print(f"Leapfrog relative energy error: {dE_lf:.6e}")

    # Both should conserve energy reasonably well
    assert dE_dop < 1e-5
    assert dE_lf < 1e-5


def test_chen_vs_fardal_df(rng, basic_hamiltonian, progenitor_w0, progenitor_mass):
    """Test that both distribution functions work with both integrators."""
    for DFClass in [ChenStreamDF, FardalStreamDF]:
        df = DFClass(random_state=rng)
        gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)

        for Integrator in [gi.DOPRI853Integrator, gi.LeapfrogIntegrator]:
            stream, prog = gen.run(
                progenitor_w0,
                progenitor_mass,
                dt=-1.0,
                n_steps=20,
                Integrator=Integrator,
            )
            assert stream.shape[0] > 0
            assert np.isfinite(stream.xyz).all()


@pytest.mark.skipif(not HAS_H5PY, reason="h5py required for this test")
def test_integrator_comparison_with_nbody(
    tmpdir, rng, basic_potential, basic_hamiltonian, progenitor_w0, progenitor_mass
):
    """Test that both integrators work correctly with N-body interactions."""
    import h5py

    df = FardalStreamDF(gala_modified=True, random_state=rng)
    gen = MockStreamGenerator(df=df, hamiltonian=basic_hamiltonian)

    # Create a perturbing N-body
    nbody_w0 = PhaseSpacePosition([20, 0, 0] * u.kpc, [0, 100, 0] * u.km / u.s)
    nbody = DirectNBody(
        w0=nbody_w0,
        external_potential=basic_potential,
        particle_potentials=[
            NFWPotential(m=1e8 * u.Msun, r_s=0.2 * u.kpc, units=galactic)
        ],
    )

    for Integrator in [
        gi.DOPRI853Integrator
    ]:  # TODO: add LeapfrogIntegrator if animation support added
        filename = os.path.join(str(tmpdir), f"nbody_{Integrator.__name__}.hdf5")

        stream, prog = gen.run(
            progenitor_w0,
            progenitor_mass,
            dt=-1.0,
            n_steps=30,
            nbody=nbody,
            Integrator=Integrator,
            output_every=5,
            output_filename=filename,
            overwrite=True,
            check_filesize=False,
        )

        # Verify output file
        with h5py.File(filename, mode="r") as f:
            stream_orbits = Orbit.from_hdf5(f["stream"])
            nbody_orbits = Orbit.from_hdf5(f["nbody"])

            # Check that nbody orbits are all finite (no NaNs)
            assert np.isfinite(nbody_orbits.xyz).all()
            assert nbody_orbits.shape[1] == 2  # progenitor + perturber

            # For stream orbits, NaNs are expected for particles not yet released
            # Just check that we have some finite values
            assert np.isfinite(
                stream_orbits[:, 0].xyz
            ).all()  # First particle should always be finite


def test_rotate_to_progenitor_plane_unit():
    """Unit test for rotate_to_progenitor_plane with manually constructed data.

    This test verifies that the rotation correctly places the progenitor at
    the origin with velocity along the x-axis, and the stream is in the xy-plane.
    """
    prog_pos = [10.0, 5.0, 3.0] * u.kpc
    prog_vel = [50.0, 100.0, 25.0] * u.km / u.s
    prog_w = PhaseSpacePosition(pos=prog_pos, vel=prog_vel)

    # a fake "stream" with particles around the progenitor
    stream_pos = (
        prog_pos[:, None]
        + np.array(
            [
                [1.0, 0.0, 0.0],  # Leading particle along x
                [-1.0, 0.0, 0.0],  # Trailing particle along x
                [0.0, 1.0, 0.0],  # Particle along y
                [0.0, 0.0, 1.0],  # Particle along z
            ]
        ).T
        * u.kpc
    )

    stream_vel = (
        prog_vel[:, None]
        + np.array(
            [
                [10.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ).T
        * u.km
        / u.s
    )

    stream = MockStream(
        pos=stream_pos,
        vel=stream_vel,
        release_time=[0.0, 1.0, 2.0, 3.0] * u.Myr,
        lead_trail=np.array([1, -1, 1, -1]),
    )

    # Rotate to xy-plane
    rotated_stream = stream.rotate_to_progenitor_plane(prog_w)

    # the transformation should preserve distances from progenitor
    # (rotation is a rigid transformation)
    original_distances = np.sqrt(np.sum((stream.xyz - prog_pos[:, None]) ** 2, axis=0))
    rotated_distances = np.sqrt(np.sum(rotated_stream.xyz**2, axis=0))
    assert u.allclose(original_distances, rotated_distances, rtol=1e-10)

    # release times and lead_trail should be preserved
    assert u.allclose(rotated_stream.release_time, stream.release_time)
    assert np.array_equal(rotated_stream.lead_trail, stream.lead_trail)

    # the stream should be centered near the origin (progenitor was translated)
    mean_pos = np.mean(rotated_stream.xyz, axis=1)
    assert np.allclose(mean_pos.value, 0.0, atol=1.0)  # Within 1 kpc of origin

    # shape should be preserved
    assert rotated_stream.shape == stream.shape


def test_rotate_to_progenitor_plane_functional(
    rng, basic_hamiltonian, progenitor_w0, progenitor_mass
):
    """Functional test for rotate_to_progenitor_plane with a real generated stream.

    This test generates a full mock stream and verifies that the rotation works
    correctly with realistic data.
    """
    # Generate a mock stream
    df = FardalStreamDF(gala_modified=True, random_state=rng)
    prog_pot = HernquistPotential(progenitor_mass, 4 * u.pc, units=galactic)

    gen = MockStreamGenerator(
        df=df, hamiltonian=basic_hamiltonian, progenitor_potential=prog_pot
    )

    # Generate stream with both leading and trailing tails
    stream, prog = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=-1.0,
        n_steps=100,
        n_particles=2,
    )

    # Get the progenitor position at the final time (same as stream)
    # Need to extract as a PhaseSpacePosition, not an Orbit slice
    prog_final = PhaseSpacePosition(pos=prog.xyz[:, -1], vel=prog.v_xyz[:, -1])

    # Rotate to xy-plane
    rotated_stream = stream.rotate_to_progenitor_plane(prog_final)

    # Verify key properties:
    # 1. Original attributes should be preserved
    assert u.allclose(rotated_stream.release_time, stream.release_time)
    assert np.array_equal(rotated_stream.lead_trail, stream.lead_trail)
    assert rotated_stream.shape == stream.shape

    # 2. Verify the transformation preserves distances from progenitor
    # (rotation + translation)
    original_distances = np.sqrt(
        np.sum((stream.xyz - prog_final.xyz[:, None]) ** 2, axis=0)
    )
    rotated_distances = np.sqrt(np.sum(rotated_stream.xyz**2, axis=0))
    assert u.allclose(original_distances, rotated_distances, rtol=1e-10)

    # 3. The stream should have particles in both +x and -x directions (lead/trail)

    # More than half of leading particles should have positive x
    n_pos_x = np.sum(rotated_stream.x[stream.lead_trail == "l"] > 0)
    assert n_pos_x > 0.5 * np.sum(stream.lead_trail == "l")

    # More than half of trailing particles should have negative x
    n_neg_x = np.sum(rotated_stream.x[stream.lead_trail == "t"] < 0)
    assert n_neg_x > 0.5 * np.sum(stream.lead_trail == "t")


# ==============================================================================
# Regression tests for progenitor final position bug
# ==============================================================================


def test_leapfrog_progenitor_final_position_uniform_release(
    basic_hamiltonian, progenitor_mass
):
    """Regression test: Leapfrog progenitor should match direct orbit integration.

    This tests the case where particles are released at every timestep.
    Tests that the progenitor position from mockstream_leapfrog matches
    a direct orbit integration with the leapfrog integrator.

    Regression test for bug where progenitor didn't end at correct position.
    """
    from gala.potential import MilkyWayPotential

    prog_w0 = PhaseSpacePosition(
        pos=[13.0, 0.0, 20.0] * u.kpc, vel=[0, 130.0, 50] * u.km / u.s
    )

    df = ChenStreamDF()
    mw = MilkyWayPotential(version="latest")
    gen = MockStreamGenerator(df, mw)

    # Short integration with particles at every timestep
    t = np.arange(0, 100.0, 1.0) * u.Myr
    n_particles = np.ones(len(t), dtype=int)  # Particle at every timestep

    stream, prog_w = gen.run(
        prog_w0,
        prog_mass=progenitor_mass,
        n_particles=n_particles,
        t=t,
        Integrator="leapfrog",
    )

    # Compare to direct orbit integration
    expected_prog_w = mw.integrate_orbit(prog_w0, t=t, Integrator="leapfrog")

    # Progenitor final position should match within numerical precision
    # Note: prog_w has shape (3, 1), need to squeeze for comparison
    assert u.allclose(
        expected_prog_w[-1].xyz, prog_w.xyz.squeeze(), rtol=1e-10, atol=1e-10 * u.kpc
    )


def test_leapfrog_progenitor_final_position_sparse_release(
    basic_hamiltonian, progenitor_mass
):
    """Regression test: Leapfrog progenitor with sparse particle release.

    This tests the critical case where particles are only released at some timesteps,
    with nstream=0 at many times (including potentially the final time).

    This was the original bug: when the last timestep had nstream=0, the progenitor
    would not be integrated to tfinal correctly.

    Regression test for bug where progenitor didn't end at correct position
    when particles were released sparsely.
    """
    from gala.potential import MilkyWayPotential

    prog_w0 = PhaseSpacePosition(
        pos=[13.0, 0.0, 20.0] * u.kpc, vel=[0, 130.0, 50] * u.km / u.s
    )

    df = ChenStreamDF()
    mw = MilkyWayPotential(version="latest")
    gen = MockStreamGenerator(df, mw)

    # Integration with particles only at some timesteps
    t = np.arange(0, 200.0, 1.0) * u.Myr
    n_particles = np.zeros(len(t), dtype=int)
    n_particles[::5] = 1  # Release particles every 5 Myr
    # Note: last timestep (t=199) has nstream=0, second-to-last (t=195) has nstream=1

    stream, prog_w = gen.run(
        prog_w0,
        prog_mass=progenitor_mass,
        n_particles=n_particles,
        t=t,
        Integrator="leapfrog",
    )

    # Compare to direct orbit integration
    expected_prog_w = mw.integrate_orbit(prog_w0, t=t, Integrator="leapfrog")

    # Progenitor final position should match within numerical precision
    # This is the key test: the progenitor must reach tfinal=199, not stop at t=195
    # Note: prog_w has shape (3, 1), need to squeeze for comparison
    assert u.allclose(
        expected_prog_w[-1].xyz, prog_w.xyz.squeeze(), rtol=1e-10, atol=1e-10 * u.kpc
    )


def test_leapfrog_vs_dop853_consistency(basic_hamiltonian, progenitor_mass):
    """Test that leapfrog and dop853 produce consistent results.

    While the integrators are different and will produce slightly different
    trajectories, they should both correctly integrate to tfinal and produce
    progenitor positions that match their respective direct orbit integrations.
    """
    from gala.potential import MilkyWayPotential

    prog_w0 = PhaseSpacePosition(
        pos=[13.0, 0.0, 20.0] * u.kpc, vel=[0, 130.0, 50] * u.km / u.s
    )

    df = ChenStreamDF()
    mw = MilkyWayPotential(version="latest")
    gen = MockStreamGenerator(df, mw)

    # Short integration with sparse particle release
    t = np.arange(0, 100.0, 1.0) * u.Myr
    n_particles = np.zeros(len(t), dtype=int)
    n_particles[::5] = 1

    # Generate with both integrators
    stream_lf, prog_w_lf = gen.run(
        prog_w0,
        prog_mass=progenitor_mass,
        n_particles=n_particles,
        t=t,
        Integrator="leapfrog",
    )

    stream_dop, prog_w_dop = gen.run(
        prog_w0,
        prog_mass=progenitor_mass,
        n_particles=n_particles,
        t=t,
        Integrator="dop853",
    )

    # Compare each to direct orbit integration with same integrator
    expected_lf = mw.integrate_orbit(prog_w0, t=t, Integrator="leapfrog")
    expected_dop = mw.integrate_orbit(prog_w0, t=t, Integrator="dop853")

    # Each should match its corresponding direct integration
    # Note: prog_w has shape (3, 1), need to squeeze for comparison
    assert u.allclose(
        expected_lf[-1].xyz, prog_w_lf.xyz.squeeze(), rtol=1e-10, atol=1e-10 * u.kpc
    )
    assert u.allclose(
        expected_dop[-1].xyz, prog_w_dop.xyz.squeeze(), rtol=1e-10, atol=1e-10 * u.kpc
    )

    # The two integrators will give different results, but they should both be
    # reasonably close (within a few kpc for this short integration)
    diff = np.linalg.norm((prog_w_lf.xyz - prog_w_dop.xyz).to_value(u.kpc))
    assert diff < 1.0  # Should differ by less than 1 kpc for this short integration
