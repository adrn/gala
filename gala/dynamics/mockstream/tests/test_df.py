# Standard library
import os

# Third-party
import astropy.units as u
import numpy as np
import pytest

# Custom
from ....integrate import DOPRI853Integrator
from ....potential import (Hamiltonian, HernquistPotential, MilkyWayPotential,
                           ConstantRotatingFrame)
from ....units import galactic
from ...core import PhaseSpacePosition

# Project
from ..df import StreaklineStreamDF, FardalStreamDF, LagrangeCloudStreamDF


# _DF_CLASSES = [StreaklineStreamDF, FardalStreamDF, LagrangeCloudStreamDF]
_DF_CLASSES = [StreaklineStreamDF]
_TEST_POTENTIALS = [HernquistPotential(m=1e12, c=5, units=galactic),
                    MilkyWayPotential()]

@pytest.mark.parametrize('DF', _DF_CLASSES)
@pytest.mark.parametrize('pot', _TEST_POTENTIALS)
def test_init_sample(DF, pot):
    H = Hamiltonian(pot)

    orbit = H.integrate_orbit([10., 0, 0, 0, 0.2, 0], dt=1., n_steps=100)
    n_times = len(orbit.t)

    # Different ways to initialize successfully:
    df = DF()
    o = df.sample(orbit, 1e4*u.Msun)
    assert len(o.x) == 2 * n_times

    df = DF(lead=False)
    o = df.sample(orbit, 1e4*u.Msun)
    assert len(o.x) == n_times

    df = DF(trail=False)
    o = df.sample(orbit, 1e4*u.Msun)
    assert len(o.x) == n_times

    df1 = DF(random_state=np.random.RandomState(42))
    o1 = df1.sample(orbit, 1e4*u.Msun)
    df2 = DF(random_state=np.random.RandomState(42))
    o2 = df2.sample(orbit, 1e4*u.Msun)
    assert u.allclose(o1.xyz, o2.xyz)
    assert u.allclose(o1.v_xyz, o2.v_xyz)
    assert len(o1.x) == 2 * n_times


@pytest.mark.parametrize('DF', _DF_CLASSES)
def test_expected_failure(DF):

    # Expected failure:
    with pytest.raises(ValueError):
        DF(lead=False, trail=False)

# @pytest.mark.parametrize('DF', _DF_CLASSES)
def test_rotating_frame():
    DF = _DF_CLASSES[0]
    H_static = Hamiltonian(_TEST_POTENTIALS[0])

    w0 = PhaseSpacePosition(pos=[10., 0, 0]*u.kpc,
                            vel=[0, 220, 0.]*u.km/u.s,
                            frame=H_static.frame)
    int_kwargs = dict(w0=w0, dt=1., n_steps=100, Integrator=DOPRI853Integrator)

    orbit_static = H_static.integrate_orbit(**int_kwargs)

    rframe = ConstantRotatingFrame([0, 0, -40] * u.km/u.s/u.kpc,
                                   units=galactic)
    H_rotating = Hamiltonian(_TEST_POTENTIALS[0],
                             frame=rframe)
    orbit_rotating = H_rotating.integrate_orbit(**int_kwargs)

    _o = orbit_rotating.to_frame(H_static.frame)
    assert u.allclose(_o.xyz, orbit_static.xyz, atol=1e-10*u.kpc)
    assert u.allclose(_o.v_xyz, orbit_static.v_xyz, atol=1e-10*u.km/u.s)

    df_static = DF()
    xvt_static = df_static.sample(orbit_static, 1e6*u.Msun)

    df_rotating = DF()
    xvt_rotating = df_rotating.sample(orbit_rotating, 1e6*u.Msun)
    xvt_rotating_static = xvt_rotating.to_frame(H_static.frame)

    assert u.allclose(xvt_static.xyz, xvt_rotating_static.xyz)
    assert u.allclose(xvt_static.v_xyz, xvt_rotating_static.v_xyz)

    return

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    axes[0].scatter(xvt_static.x, xvt_static.y, s=10)
    axes[1].scatter(xvt_rotating.x, xvt_rotating.y, s=10)
    axes[2].scatter(xvt_rotating_static.x, xvt_rotating_static.y, s=10)
    fig.tight_layout()

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    # axes[0].scatter(p1.v_x, p1.v_y, s=5)
    # axes[1].scatter(p2.v_x, p2.v_y, s=5)
    # fig.tight_layout()

    plt.show()
