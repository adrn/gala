import os

# Third-party
import astropy.units as u
import numpy as np
import pytest

# Custom
from ....potential import Hamiltonian, NFWPotential, HernquistPotential
from ....dynamics import PhaseSpacePosition
from ....integrate import DOPRI853Integrator
from ....units import galactic

# Project
from ..core import mock_stream, streakline_stream, fardal_stream


def test_mock_stream():
    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.,
                                                    units=galactic)
    ham = Hamiltonian(potential)

    w0 = PhaseSpacePosition(pos=[0.,15.,0]*u.kpc,
                            vel=[-0.13,0,0]*u.kpc/u.Myr)
    prog = ham.integrate_orbit(w0, dt=-2., n_steps=1023)
    prog = prog[::-1]

    k_mean = [1.,0.,0.,0.,1.,0.]
    k_disp = [0.,0.,0.,0.,0.,0.]

    with pytest.raises(NotImplementedError):
        stream = mock_stream(ham, prog, k_mean=k_mean, k_disp=k_disp,
                             prog_mass=1E4)


mock_funcs = [streakline_stream, fardal_stream]
all_extra_args = [dict(), dict()]
@pytest.mark.parametrize("mock_func, extra_kwargs", zip(mock_funcs, all_extra_args))
def test_each_type(mock_func, extra_kwargs):
    # TODO: remove this test when these functions are removed
    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.,
                                                    units=galactic)
    ham = Hamiltonian(potential)

    w0 = PhaseSpacePosition(pos=[0., 15., 0]*u.kpc,
                            vel=[-0.13, 0, 0]*u.kpc/u.Myr)
    prog = ham.integrate_orbit(w0, dt=-2., n_steps=1023)
    prog = prog[::-1]

    with pytest.warns(DeprecationWarning):
        stream = mock_func(ham, prog_orbit=prog, prog_mass=1E4*u.Msun,
                           Integrator=DOPRI853Integrator, **extra_kwargs)

    assert prog.t.shape == (1024,)
    assert stream.pos.shape == (2048,) # two particles per step


@pytest.mark.skipif('CI' in os.environ,
                    reason="For some reason, doesn't work on Travis/CI")
def test_animate(tmpdir):

    np.random.seed(42)
    pot = HernquistPotential(m=1E11, c=1., units=galactic)
    w0 = PhaseSpacePosition(pos=[5., 0, 0]*u.kpc, vel=[0, 0.1, 0]*u.kpc/u.Myr)
    orbit = Hamiltonian(pot).integrate_orbit(w0, dt=1., n_steps=1000,
                                             Integrator=DOPRI853Integrator)

    fardal_stream(pot, orbit, prog_mass=1E5*u.Msun, release_every=10,
                  snapshot_filename=os.path.join(str(tmpdir), "test.hdf5"),
                  seed=42)

    stream = fardal_stream(pot, orbit, prog_mass=1E5*u.Msun, release_every=10,
                           seed=42)

    import h5py
    with h5py.File(os.path.join(str(tmpdir), "test.hdf5")) as f:
        t = f['t'][:]
        pos = f['pos'][:]
        vel = f['vel'][:]

    assert np.allclose(t, orbit.t.value)

    for idx in range(pos.shape[2]):
        assert np.allclose(pos[:, -1, idx], stream.xyz.value[:, idx], rtol=1E-4)
        assert np.allclose(vel[:, -1, idx], stream.v_xyz.value[:, idx], rtol=1E-4)


@pytest.mark.skipif('CI' in os.environ,
                    reason="For some reason, doesn't work on Travis/CI")
def test_animate_output_every(tmpdir):
    snapshot_filename = os.path.join(str(tmpdir), "test.hdf5")

    np.random.seed(42)
    pot = HernquistPotential(m=1E11, c=1., units=galactic)
    w0 = PhaseSpacePosition(pos=[5., 0, 0]*u.kpc, vel=[0, 0.1, 0]*u.kpc/u.Myr)
    orbit = Hamiltonian(pot).integrate_orbit(w0, dt=1., n_steps=100,
                                             Integrator=DOPRI853Integrator)

    fardal_stream(pot, orbit, prog_mass=1E5*u.Msun, release_every=10,
                  snapshot_filename=snapshot_filename,
                  output_every=20, seed=42)

    stream = fardal_stream(pot, orbit, prog_mass=1E5*u.Msun, release_every=10,
                           seed=42)

    import h5py
    with h5py.File(snapshot_filename, 'r') as f:
        t = f['t'][:]
        pos = f['pos'][:]
        vel = f['vel'][:]

    assert np.allclose(t, orbit.t.value)

    for idx in range(pos.shape[2]):
        assert np.allclose(pos[:, -1, idx], stream.xyz.value[:, idx], rtol=1E-4)
        assert np.allclose(vel[:, -1, idx], stream.v_xyz.value[:, idx], rtol=1E-4)

    assert pos.shape[1] == (100 // 20 + 2) # initial and final
