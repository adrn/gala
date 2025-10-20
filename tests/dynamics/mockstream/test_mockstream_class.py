import astropy.units as u
import numpy as np
import pytest

from gala.dynamics.core import PhaseSpacePosition
from gala.dynamics.mockstream import MockStream


def test_init():
    xyz = np.random.random(size=(3, 100)) * u.kpc
    vxyz = np.random.random(size=(3, 100)) * u.km / u.s
    t1 = np.random.random(size=100) * u.Myr

    lead_trail = np.empty(100, dtype="U1")
    lead_trail[::2] = "t"
    lead_trail[1::2] = "l"

    stream = MockStream(xyz, vxyz)
    stream = MockStream(xyz, vxyz, release_time=t1)
    stream = MockStream(xyz, vxyz, lead_trail=lead_trail)

    with pytest.raises(ValueError):
        MockStream(xyz, vxyz, release_time=t1[:-1])

    with pytest.raises(ValueError):
        MockStream(xyz, vxyz, lead_trail=lead_trail[:-1])


def test_no_copy():
    xyz = np.random.random(size=(3, 100)) * u.kpc
    vxyz = np.random.random(size=(3, 100)) * u.km / u.s

    s1 = MockStream(xyz, vxyz, copy=True)
    s2 = MockStream(xyz, vxyz, copy=False)

    xyz[0, 0] = 999.0 * u.kpc
    assert s1.pos[0].x.value != 999.0
    assert s2.pos[0].x.value == 999.0


def test_one_burst():
    # Regression test: Tests a bug found by Helmer when putting all particles at
    # one timestep
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.dynamics import mockstream as ms
    from gala.units import galactic

    # NFW MW with v_c = 232.8 km/s @ r = 8.2 kpc
    pot = gp.NFWPotential.from_circular_velocity(
        v_c=232.8 * u.km / u.s, r_s=15 * u.kpc, r_ref=8.2 * u.kpc, units=galactic
    )

    H = gp.Hamiltonian(pot)

    prog_w0 = gd.PhaseSpacePosition(
        pos=[10, 0, 0.0] * u.kpc, vel=[0, 10, 0.0] * u.km / u.s
    )

    dt = 1 * u.Myr
    nsteps = 100
    orbit = H.integrate_orbit(prog_w0, dt=dt, n_steps=nsteps)

    r = orbit.spherical.distance

    n_array = np.zeros(orbit.t.size, dtype=int)
    argmin = r[0:150].argmin()
    n_array[argmin] = 1000

    df = ms.FardalStreamDF(
        gala_modified=True, random_state=np.random.default_rng(seed=42)
    )

    dt = 1 * u.Myr
    prog_mass = 2.5e4 * u.Msun
    prog_pot = gp.PlummerPotential(m=prog_mass, b=4 * u.pc, units=galactic)

    gen = ms.MockStreamGenerator(df, H, progenitor_potential=prog_pot)

    stream, prog = gen.run(
        prog_w0, prog_mass, n_particles=n_array, dt=dt, n_steps=nsteps, progress=False
    )

    # Sanity check the first stream particle and the progenitor
    stream0_true = PhaseSpacePosition(
        pos=[-10.07444187, -1.37424641, 0.06310397] * u.kpc,
        vel=[-0.05672946, -0.01837671, 0.00038504] * u.kpc / u.Myr,
    )
    prog_true = PhaseSpacePosition(
        pos=[-9.72388107, -1.28632464, 0.0] * u.kpc,
        vel=[-0.04714419, -0.016754, 0.0] * u.kpc / u.Myr,
    )

    assert u.allclose(stream[0].xyz, stream0_true.xyz)
    assert u.allclose(stream[0].v_xyz, stream0_true.v_xyz)
    assert u.allclose(prog[0].xyz, prog_true.xyz)
    assert u.allclose(prog[0].v_xyz, prog_true.v_xyz)


def test_Fardal_vs_GalaModified():
    """
    Regression test: Check that one can actually use the original Fardal parameter
    values, and that makes a different stream than the Gala-modified values:
    https://github.com/adrn/gala/pull/358
    """
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.dynamics import mockstream as ms
    from gala.units import galactic

    # NFW MW with v_c = 232.8 km/s @ r = 8.2 kpc
    pot = gp.NFWPotential.from_circular_velocity(
        v_c=232.8 * u.km / u.s, r_s=15 * u.kpc, r_ref=8.2 * u.kpc, units=galactic
    )

    H = gp.Hamiltonian(pot)

    prog_w0 = gd.PhaseSpacePosition(
        pos=[10, 0, 0.0] * u.kpc, vel=[0, 300, 20.0] * u.km / u.s
    )

    with pytest.warns(FutureWarning, match="Fardal"):
        ms.FardalStreamDF()

    df_false = ms.FardalStreamDF(
        gala_modified=False, random_state=np.random.default_rng(seed=42)
    )
    df_true = ms.FardalStreamDF(
        gala_modified=True, random_state=np.random.default_rng(seed=42)
    )

    gen_false = ms.MockStreamGenerator(df_false, H)
    gen_true = ms.MockStreamGenerator(df_true, H)

    prog_mass = 2.5e4 * u.Msun
    stream_false, _ = gen_false.run(
        prog_w0, prog_mass, dt=1, n_steps=128, progress=False
    )
    stream_true, _ = gen_true.run(prog_w0, prog_mass, dt=1, n_steps=128, progress=False)

    assert not u.allclose(stream_false.xyz, stream_true.xyz)
