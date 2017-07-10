# coding: utf-8

from __future__ import division, print_function

import os
import warnings

# Third-party
import astropy.units as u
import numpy as np
import pytest

# Custom
from ....potential import Hamiltonian, NFWPotential, HernquistPotential
from ....dynamics import PhaseSpacePosition
from ....integrate import DOPRI853Integrator, LeapfrogIntegrator, RK5Integrator
from ....units import galactic

# Project
from ..core import mock_stream, streakline_stream, fardal_stream, dissolved_fardal_stream

@pytest.mark.parametrize("Integrator,kwargs",
                         zip([DOPRI853Integrator, LeapfrogIntegrator],
                             [dict(), dict()]))
def test_mock_stream(Integrator, kwargs):
    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.,
                                                    units=galactic)
    ham = Hamiltonian(potential)

    w0 = PhaseSpacePosition(pos=[0.,15.,0]*u.kpc,
                            vel=[-0.13,0,0]*u.kpc/u.Myr)
    prog = ham.integrate_orbit(w0, dt=-2., n_steps=1023)
    prog = prog[::-1]

    k_mean = [1.,0.,0.,0.,1.,0.]
    k_disp = [0.,0.,0.,0.,0.,0.]
    stream = mock_stream(ham, prog, k_mean=k_mean, k_disp=k_disp,
                         prog_mass=1E4, Integrator=Integrator, **kwargs)

    # fig = prog.plot(subplots_kwargs=dict(sharex=False,sharey=False))
    # fig = stream.plot()

    # fig = prog.plot(subplots_kwargs=dict(sharex=False,sharey=False))
    # fig = stream.plot(color='#ff0000', alpha=0.5, axes=fig.axes)

    # pl.show()
    # return

    assert stream.pos.shape == (2048,) # two particles per step

    diff = np.abs(stream[-2:].xyz - prog[-1:].xyz)
    assert np.allclose(diff[0].value, 0.)
    assert np.allclose(diff[1,0].value, diff[1,1].value)
    assert np.allclose(diff[2].value, 0.)

mock_funcs = [streakline_stream, fardal_stream, dissolved_fardal_stream]
all_extra_args = [dict(), dict(), dict(t_disrupt=-250.*u.Myr)]
@pytest.mark.parametrize("mock_func, extra_kwargs", zip(mock_funcs, all_extra_args))
def test_each_type(mock_func, extra_kwargs):
    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.,
                                                    units=galactic)
    ham = Hamiltonian(potential)

    w0 = PhaseSpacePosition(pos=[0.,15.,0]*u.kpc,
                            vel=[-0.13,0,0]*u.kpc/u.Myr)
    prog = ham.integrate_orbit(w0, dt=-2., n_steps=1023)
    prog = prog[::-1]

    stream = mock_func(ham, prog_orbit=prog, prog_mass=1E4,
                       Integrator=DOPRI853Integrator, **extra_kwargs)

    # import matplotlib.pyplot as plt
    # fig = prog.plot(subplots_kwargs=dict(sharex=False,sharey=False))
    # fig = stream.plot(color='#ff0000', alpha=0.5, axes=fig.axes)
    # fig = stream.plot()
    # plt.show()

    assert prog.t.shape == (1024,)
    assert stream.pos.shape == (2048,) # two particles per step

    # -----------------------
    # Test expected failures:

    # Deprecation warning for passing in potential
    warnings.simplefilter('always')
    with pytest.warns(DeprecationWarning):
        stream = mock_func(potential, prog_orbit=prog, prog_mass=1E4*u.Msun,
                           Integrator=DOPRI853Integrator, **extra_kwargs)

    # Integrator not supported
    with pytest.raises(ValueError):
        stream = mock_func(potential, prog_orbit=prog, prog_mass=1E4*u.Msun,
                           Integrator=RK5Integrator, **extra_kwargs)

    # Passed a phase-space position, not orbit
    with pytest.raises(TypeError):
        stream = mock_func(potential, prog_orbit=prog[0], prog_mass=1E4*u.Msun,
                           Integrator=DOPRI853Integrator, **extra_kwargs)

@pytest.mark.skipif('CI' in os.environ,
                    reason="For some reason, doesn't work on Travis/CI")
def test_animate(tmpdir):

    np.random.seed(42)
    pot = HernquistPotential(m=1E11, c=1., units=galactic)
    w0 = PhaseSpacePosition(pos=[5.,0,0]*u.kpc, vel=[0,0.1,0]*u.kpc/u.Myr)
    orbit = pot.integrate_orbit(w0, dt=1., n_steps=1000,
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
        assert np.allclose(pos[:,-1,idx], stream.xyz.value[:,idx], rtol=1E-4)
        assert np.allclose(vel[:,-1,idx], stream.v_xyz.value[:,idx], rtol=1E-4)
