import astropy.units as u
import numpy as np
import pytest

import gala.dynamics as gd
import gala.dynamics.mockstream as ms
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

_DF_CLASSES = [
    ms.StreaklineStreamDF,
    ms.FardalStreamDF,
    ms.LagrangeCloudStreamDF,
    ms.ChenStreamDF,
]
_DF_KWARGS = [{}, {"gala_modified": True}, {"v_disp": 1 * u.km / u.s}]
_TEST_POTENTIALS = [
    gp.HernquistPotential(m=1e12, c=5, units=galactic),
    gp.MilkyWayPotential(),
]


@pytest.mark.parametrize(("DF", "DF_kwargs"), zip(_DF_CLASSES, _DF_KWARGS))
@pytest.mark.parametrize("pot", _TEST_POTENTIALS)
def test_init_sample(DF, DF_kwargs, pot):
    H = gp.Hamiltonian(pot)

    orbit = H.integrate_orbit([10.0, 0, 0, 0, 0.2, 0], dt=1.0, n_steps=100)
    n_times = len(orbit.t)

    # Different ways to initialize successfully:
    df = DF(**DF_kwargs)
    o = df.sample(orbit, 1e4 * u.Msun)
    assert len(o.x) == 2 * n_times

    df = DF(lead=False, **DF_kwargs)
    o = df.sample(orbit, 1e4 * u.Msun)
    assert len(o.x) == n_times

    df = DF(trail=False, **DF_kwargs)
    o = df.sample(orbit, 1e4 * u.Msun)
    assert len(o.x) == n_times

    df1 = DF(random_state=np.random.RandomState(42), **DF_kwargs)
    o1 = df1.sample(orbit, 1e4 * u.Msun)
    df2 = DF(random_state=np.random.RandomState(42), **DF_kwargs)
    o2 = df2.sample(orbit, 1e4 * u.Msun)
    assert u.allclose(o1.xyz, o2.xyz)
    assert u.allclose(o1.v_xyz, o2.v_xyz)
    assert len(o1.x) == 2 * n_times


@pytest.mark.parametrize(("DF", "DF_kwargs"), zip(_DF_CLASSES, _DF_KWARGS))
def test_expected_failure(DF, DF_kwargs):
    # Expected failure:
    with pytest.raises(ValueError):
        DF(lead=False, trail=False, **DF_kwargs)


def test_rotating_frame():
    DF = _DF_CLASSES[0]
    H_static = gp.Hamiltonian(_TEST_POTENTIALS[0])

    w0 = gd.PhaseSpacePosition(
        pos=[10.0, 0, 0] * u.kpc, vel=[0, 220, 0.0] * u.km / u.s, frame=H_static.frame
    )
    int_kwargs = {
        "w0": w0,
        "dt": 1,
        "n_steps": 100,
        "Integrator": gi.DOPRI853Integrator,
    }

    orbit_static = H_static.integrate_orbit(**int_kwargs)

    rframe = gp.ConstantRotatingFrame([0, 0, -40] * u.km / u.s / u.kpc, units=galactic)
    H_rotating = gp.Hamiltonian(_TEST_POTENTIALS[0], frame=rframe)
    orbit_rotating = H_rotating.integrate_orbit(**int_kwargs)

    o = orbit_rotating.to_frame(H_static.frame)
    assert u.allclose(o.xyz, orbit_static.xyz, atol=1e-13 * u.kpc)
    assert u.allclose(o.v_xyz, orbit_static.v_xyz, atol=1e-13 * u.km / u.s)

    df_static = DF(trail=False)
    xvt_static = df_static.sample(orbit_static, 1e6 * u.Msun)

    df_rotating = DF(trail=False)
    xvt_rotating = df_rotating.sample(orbit_rotating, 1e6 * u.Msun)
    xvt_rotating_static = xvt_rotating.to_frame(
        H_static.frame, t=xvt_rotating.release_time
    )

    assert u.allclose(xvt_static.xyz, xvt_rotating_static.xyz, atol=1e-9 * u.kpc)
    assert u.allclose(
        xvt_static.v_xyz, xvt_rotating_static.v_xyz, atol=1e-9 * u.kpc / u.Myr
    )


def test_regression_415():
    """
    Regression test for #415: integration error when progenitor mass is an array with
    length 1 when it should be a scalar.
    """
    pot = gp.NFWPotential(1e12 * u.Msun, r_s=10.0 * u.kpc, units=galactic)
    w0 = gd.PhaseSpacePosition(
        pos=[39.54522670882826, -21.408405557971204, 67.2661672] * u.kpc,
        vel=[-1.87539782e02, -3.59878933e02, 1.08545075e02] * u.km / u.s,
    )
    progenitor_mass = np.array([10**8]) * u.Msun
    dw_pot = gp.PlummerPotential(m=progenitor_mass[0], b=50 * u.pc, units=galactic)
    df = ms.ChenStreamDF(random_state=np.random.default_rng(42))
    gen_pal5 = ms.MockStreamGenerator(df, pot, progenitor_potential=dw_pot)
    xorbit = pot.integrate_orbit(w0, dt=-1 * u.Myr, n_steps=100)
    w0 = gd.PhaseSpacePosition(pos=xorbit.pos[-1], vel=xorbit.vel[-1])
    xnsteps = 1000
    stream, _ = gen_pal5.run(
        w0, progenitor_mass, dt=1 * u.Myr, n_steps=xnsteps, n_particles=1
    )

    assert stream.shape == (2002,)
