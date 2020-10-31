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


_DF_CLASSES = [StreaklineStreamDF, FardalStreamDF, LagrangeCloudStreamDF]
_DF_KWARGS = [{}, {}, {'v_disp': 1*u.km/u.s}]
_TEST_POTENTIALS = [HernquistPotential(m=1e12, c=5, units=galactic),
                    MilkyWayPotential()]


@pytest.mark.parametrize('DF, DF_kwargs', zip(_DF_CLASSES, _DF_KWARGS))
@pytest.mark.parametrize('pot', _TEST_POTENTIALS)
def test_init_sample(DF, DF_kwargs, pot):
    H = Hamiltonian(pot)

    orbit = H.integrate_orbit([10., 0, 0, 0, 0.2, 0], dt=1., n_steps=100)
    n_times = len(orbit.t)

    # Different ways to initialize successfully:
    df = DF(**DF_kwargs)
    o = df.sample(orbit, 1e4*u.Msun)
    assert len(o.x) == 2 * n_times

    df = DF(lead=False, **DF_kwargs)
    o = df.sample(orbit, 1e4*u.Msun)
    assert len(o.x) == n_times

    df = DF(trail=False, **DF_kwargs)
    o = df.sample(orbit, 1e4*u.Msun)
    assert len(o.x) == n_times

    df1 = DF(random_state=np.random.RandomState(42), **DF_kwargs)
    o1 = df1.sample(orbit, 1e4*u.Msun)
    df2 = DF(random_state=np.random.RandomState(42), **DF_kwargs)
    o2 = df2.sample(orbit, 1e4*u.Msun)
    assert u.allclose(o1.xyz, o2.xyz)
    assert u.allclose(o1.v_xyz, o2.v_xyz)
    assert len(o1.x) == 2 * n_times


@pytest.mark.parametrize('DF, DF_kwargs', zip(_DF_CLASSES, _DF_KWARGS))
def test_expected_failure(DF, DF_kwargs):

    # Expected failure:
    with pytest.raises(ValueError):
        DF(lead=False, trail=False, **DF_kwargs)


def test_rotating_frame():
    DF = _DF_CLASSES[0]
    H_static = Hamiltonian(_TEST_POTENTIALS[0])

    w0 = PhaseSpacePosition(pos=[10., 0, 0]*u.kpc,
                            vel=[0, 220, 0.]*u.km/u.s,
                            frame=H_static.frame)
    int_kwargs = dict(w0=w0, dt=1, n_steps=100,
                      Integrator=DOPRI853Integrator)

    orbit_static = H_static.integrate_orbit(**int_kwargs)

    rframe = ConstantRotatingFrame([0, 0, -40] * u.km/u.s/u.kpc,
                                   units=galactic)
    H_rotating = Hamiltonian(_TEST_POTENTIALS[0],
                             frame=rframe)
    orbit_rotating = H_rotating.integrate_orbit(**int_kwargs)

    _o = orbit_rotating.to_frame(H_static.frame)
    assert u.allclose(_o.xyz, orbit_static.xyz, atol=1e-13*u.kpc)
    assert u.allclose(_o.v_xyz, orbit_static.v_xyz, atol=1e-13*u.km/u.s)

    df_static = DF(trail=False)
    xvt_static = df_static.sample(orbit_static, 1e6*u.Msun)

    df_rotating = DF(trail=False)
    xvt_rotating = df_rotating.sample(orbit_rotating, 1e6*u.Msun)
    xvt_rotating_static = xvt_rotating.to_frame(H_static.frame,
                                                t=xvt_rotating.release_time)

    assert u.allclose(xvt_static.xyz, xvt_rotating_static.xyz,
                      atol=1e-9*u.kpc)
    assert u.allclose(xvt_static.v_xyz, xvt_rotating_static.v_xyz,
                      atol=1e-9*u.kpc/u.Myr)
