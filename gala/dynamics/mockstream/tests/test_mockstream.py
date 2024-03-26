import itertools
import os

# Third-party
import astropy.units as u
import numpy as np
import pytest

from gala.tests.optional_deps import HAS_H5PY

from ....dynamics import Orbit, PhaseSpacePosition
from ....potential import (
    ConstantRotatingFrame,
    Hamiltonian,
    HernquistPotential,
    NFWPotential,
)
from ....units import galactic
from ...nbody import DirectNBody
from ..df import FardalStreamDF
from ..mockstream_generator import MockStreamGenerator


def test_init():
    w0 = PhaseSpacePosition(
        pos=[15.0, 0.0, 0] * u.kpc, vel=[0, 0, 0.13] * u.kpc / u.Myr
    )
    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.0, units=galactic)
    H = Hamiltonian(potential)
    df = FardalStreamDF(gala_modified=True)

    with pytest.raises(TypeError):
        MockStreamGenerator(df="some df", hamiltonian=H)

    with pytest.raises(TypeError):
        MockStreamGenerator(df=df, hamiltonian=H, progenitor_potential="stuff")

    # Test validating the input nbody
    nbody_w0 = PhaseSpacePosition(
        pos=[25.0, 0.0, 0] * u.kpc, vel=[0, 0, 0.13] * u.kpc / u.Myr
    )
    potential2 = NFWPotential.from_circular_velocity(v_c=0.2, r_s=25.0, units=galactic)
    nbody = DirectNBody(
        w0=nbody_w0, external_potential=potential2, particle_potentials=[None]
    )
    gen = MockStreamGenerator(df=df, hamiltonian=H)
    with pytest.raises(ValueError):
        gen._get_nbody(w0, nbody)

    frame2 = ConstantRotatingFrame([0, 0, 25.0] * u.km / u.s / u.kpc, units=galactic)
    nbody = DirectNBody(
        w0=nbody_w0,
        external_potential=potential,
        frame=frame2,
        particle_potentials=[None],
    )
    with pytest.raises(ValueError):
        gen._get_nbody(w0, nbody)

    # we expect success!
    nbody = DirectNBody(
        w0=nbody_w0, external_potential=potential, particle_potentials=[None]
    )
    new_nbody = gen._get_nbody(w0, nbody)  # noqa


def test_run():
    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.0, units=galactic)
    H = Hamiltonian(potential)
    w0 = PhaseSpacePosition(
        pos=[15.0, 0.0, 0] * u.kpc, vel=[0, 0, 0.13] * u.kpc / u.Myr
    )
    mass = 2.5e4 * u.Msun
    prog_pot = HernquistPotential(mass, 4 * u.pc, units=galactic)

    # The basic run:
    df = FardalStreamDF(gala_modified=True)
    gen = MockStreamGenerator(df=df, hamiltonian=H)
    stream1, _ = gen.run(w0, mass, dt=-1.0, n_steps=100)

    # Expected errors:
    with pytest.raises(TypeError):
        gen.run(w0, mass.value, dt=-1.0, n_steps=100)

    # With self-gravity
    gen = MockStreamGenerator(df=df, hamiltonian=H, progenitor_potential=prog_pot)
    stream2, _ = gen.run(w0, mass, dt=-1.0, n_steps=100)
    assert not u.allclose(stream1.xyz, stream2.xyz)

    # Skipping release steps:
    gen = MockStreamGenerator(df=df, hamiltonian=H)
    stream3, _ = gen.run(w0, mass, dt=-1.0, n_steps=100, release_every=4, n_particles=4)
    assert stream3.shape == ((100 // 4 + 1) * 4 * 2,)

    # Custom n_particles:
    gen = MockStreamGenerator(df=df, hamiltonian=H)
    n_particles = np.random.randint(0, 4, size=101)
    stream3, _ = gen.run(
        w0, mass, dt=-1.0, n_steps=100, release_every=1, n_particles=n_particles
    )
    assert stream3.shape[0] == 2 * n_particles.sum()


@pytest.mark.parametrize("dt", [1, -1])
@pytest.mark.parametrize("save_all", [True, False])
def test_mockstream_nbody_run(dt, save_all):
    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.0, units=galactic)
    H = Hamiltonian(potential)
    w0 = PhaseSpacePosition(
        pos=[15.0, 0.0, 0] * u.kpc, vel=[0, 0, 0.13] * u.kpc / u.Myr
    )
    mass = 2.5e4 * u.Msun
    df = FardalStreamDF(gala_modified=True)

    # Test passing custom N-body:
    nbody_w0 = PhaseSpacePosition([20, 0, 0] * u.kpc, [0, 100, 0] * u.km / u.s)
    nbody = DirectNBody(
        w0=nbody_w0,
        external_potential=potential,
        particle_potentials=[
            NFWPotential(m=1e8 * u.Msun, r_s=0.2 * u.kpc, units=galactic)
        ],
        save_all=save_all,
    )
    gen = MockStreamGenerator(df=df, hamiltonian=H)
    gen.run(w0, mass, dt=dt, n_steps=100, nbody=nbody)


@pytest.mark.parametrize(
    "dt, nsteps, output_every, release_every, n_particles, trail",
    list(itertools.product([1, -1], [16, 17], [1, 2], [1, 4], [1, 4], [True, False])),
)
@pytest.mark.skipif(not HAS_H5PY, reason="h5py required for this test")
def test_animate(tmpdir, dt, nsteps, output_every, release_every, n_particles, trail):
    import h5py

    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.0, units=galactic)
    H = Hamiltonian(potential)
    w0 = PhaseSpacePosition(
        pos=[15.0, 0.0, 0] * u.kpc, vel=[0, 0, 0.13] * u.kpc / u.Myr
    )
    mass = 2.5e4 * u.Msun

    # The basic run:
    df = FardalStreamDF(gala_modified=True, trail=trail)
    gen = MockStreamGenerator(df=df, hamiltonian=H)

    filename = os.path.join(str(tmpdir), "test.hdf5")
    stream, _ = gen.run(
        w0,
        mass,
        dt=dt,
        n_steps=nsteps,
        release_every=release_every,
        n_particles=n_particles,
        output_every=output_every,
        output_filename=filename,
        overwrite=True,
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
