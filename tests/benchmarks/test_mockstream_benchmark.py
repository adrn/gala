import astropy.units as u
import numpy as np
import pytest

import gala.dynamics as gd
import gala.potential as gp
from gala.dynamics import mockstream as ms


@pytest.fixture
def rng():
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def hamiltonian():
    """Standard NFW potential for benchmarks."""
    return gp.Hamiltonian(
        gp.NFWPotential.from_circular_velocity(
            v_c=220 * u.km / u.s, r_s=15 * u.kpc, units="galactic"
        )
    )


@pytest.fixture
def progenitor_w0():
    """Standard progenitor initial conditions."""
    return gd.PhaseSpacePosition(
        pos=[10.0, 0, 0.0] * u.kpc, vel=[0, 170, 0.0] * u.km / u.s
    )


@pytest.fixture
def progenitor_mass():
    """Standard progenitor mass."""
    return 2.5e4 * u.Msun


@pytest.mark.benchmark
@pytest.mark.parametrize("StreamDF", ["FardalStreamDF", "ChenStreamDF"])
@pytest.mark.parametrize("n_steps", [10, 100, 1_000])
@pytest.mark.parametrize("n_particles", [1, 2, 10])
def test_mockstream_generation(
    hamiltonian, progenitor_w0, progenitor_mass, rng, StreamDF, n_steps, n_particles
):
    """Benchmark basic stream generation with different DFs and parameters."""
    if StreamDF == "FardalStreamDF":
        df = ms.FardalStreamDF(gala_modified=True, random_state=rng)
    elif StreamDF == "ChenStreamDF":
        df = ms.ChenStreamDF(random_state=rng)

    gen = ms.MockStreamGenerator(df=df, hamiltonian=hamiltonian)
    stream, prog = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=1 * u.Myr,
        n_steps=n_steps,
        n_particles=n_particles,
        progress=False,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("Integrator", ["dop853", "leapfrog"])
@pytest.mark.parametrize("n_steps", [10, 100, 1_000])
def test_mockstream_integrators(
    hamiltonian, progenitor_w0, progenitor_mass, rng, Integrator, n_steps
):
    """Benchmark stream generation with different integrators."""
    df = ms.FardalStreamDF(gala_modified=True, random_state=rng)
    gen = ms.MockStreamGenerator(df=df, hamiltonian=hamiltonian)

    stream, prog = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=1 * u.Myr,
        n_steps=n_steps,
        Integrator=Integrator,
        progress=False,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("release_every", [1, 5, 10])
@pytest.mark.parametrize("n_steps", [100, 1_000])
def test_mockstream_release_every(
    hamiltonian, progenitor_w0, progenitor_mass, rng, release_every, n_steps
):
    """Benchmark stream generation with different release frequencies."""
    df = ms.FardalStreamDF(gala_modified=True, random_state=rng)
    gen = ms.MockStreamGenerator(df=df, hamiltonian=hamiltonian)

    stream, prog = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=1 * u.Myr,
        n_steps=n_steps,
        release_every=release_every,
        progress=False,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n_steps", [100, 1_000])
def test_mockstream_with_self_gravity(
    hamiltonian, progenitor_w0, progenitor_mass, rng, n_steps
):
    """Benchmark stream generation with progenitor self-gravity."""
    df = ms.FardalStreamDF(gala_modified=True, random_state=rng)
    prog_pot = gp.PlummerPotential(m=progenitor_mass, b=4 * u.pc, units="galactic")
    gen = ms.MockStreamGenerator(
        df=df, hamiltonian=hamiltonian, progenitor_potential=prog_pot
    )

    stream, prog = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=1 * u.Myr,
        n_steps=n_steps,
        progress=False,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("n_steps", [100, 1_000])
@pytest.mark.parametrize("n_perturbers", [1, 3, 5])
def test_mockstream_with_nbody(
    hamiltonian, progenitor_w0, progenitor_mass, rng, n_steps, n_perturbers
):
    """Benchmark stream generation with N-body perturbers."""
    from gala.dynamics.nbody import DirectNBody

    df = ms.FardalStreamDF(gala_modified=True, random_state=rng)
    gen = ms.MockStreamGenerator(df=df, hamiltonian=hamiltonian)

    # Create perturbers with random positions and velocities
    perturber_positions = rng.uniform(-20, 20, size=(3, n_perturbers)) * u.kpc
    perturber_velocities = rng.uniform(-100, 100, size=(3, n_perturbers)) * u.km / u.s
    perturber_w0 = gd.PhaseSpacePosition(
        pos=perturber_positions, vel=perturber_velocities
    )

    # Create DirectNBody with perturbers
    particle_potentials = [
        gp.PlummerPotential(m=1e9 * u.Msun, b=100 * u.pc, units="galactic")
        for _ in range(n_perturbers)
    ]
    nbody = DirectNBody(
        w0=perturber_w0,
        particle_potentials=particle_potentials,
        external_potential=hamiltonian.potential,
        frame=hamiltonian.frame,
    )

    stream, prog = gen.run(
        progenitor_w0,
        progenitor_mass,
        dt=1 * u.Myr,
        n_steps=n_steps,
        nbody=nbody,
        progress=False,
    )
