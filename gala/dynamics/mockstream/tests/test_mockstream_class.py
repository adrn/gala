# Third-party
import astropy.units as u
import numpy as np
import pytest

# Custom
from ..core import MockStream


def test_init():

    xyz = np.random.random(size=(3, 100)) * u.kpc
    vxyz = np.random.random(size=(3, 100)) * u.km / u.s
    t1 = np.random.random(size=100) * u.Myr

    lead_trail = np.empty(100, dtype="U1")
    lead_trail[::2] = "t"
    lead_trail[1::2] = "l"

    stream = MockStream(xyz, vxyz)
    stream = MockStream(xyz, vxyz, release_time=t1)
    stream = MockStream(xyz, vxyz, lead_trail=lead_trail)  # noqa

    with pytest.raises(ValueError):
        MockStream(xyz, vxyz, release_time=t1[:-1])

    with pytest.raises(ValueError):
        MockStream(xyz, vxyz, lead_trail=lead_trail[:-1])


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

    df = ms.FardalStreamDF(gala_modified=True)

    dt = 1 * u.Myr
    prog_mass = 2.5e4 * u.Msun
    prog_pot = gp.PlummerPotential(m=prog_mass, b=4 * u.pc, units=galactic)

    gen = ms.MockStreamGenerator(df, H, progenitor_potential=prog_pot)

    stream, prog = gen.run(
        prog_w0, prog_mass, n_particles=n_array, dt=dt, n_steps=nsteps, progress=False
    )


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

    with pytest.warns(DeprecationWarning, match="Fardal"):
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
