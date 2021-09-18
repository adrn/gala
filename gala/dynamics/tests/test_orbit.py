# Third-party
from astropy.coordinates import (SphericalRepresentation, Galactic,
                                 SphericalCosLatDifferential)
import astropy.units as u
import numpy as np
import pytest
import scipy.optimize as so

# Project
from ..core import PhaseSpacePosition
from ..orbit import Orbit
from ...integrate import DOPRI853Integrator
from ...potential import (Hamiltonian, HernquistPotential, LogarithmicPotential,
                          KeplerPotential)
from ...potential.frame import StaticFrame, ConstantRotatingFrame
from ...units import galactic, solarsystem
from ..util import combine
from gala.tests.optional_deps import HAS_H5PY, HAS_GALPY


# Tests below should be cleaned up a bit...
def test_initialize():

    with pytest.raises(ValueError):
        x = np.random.random(size=(3, 10))
        v = np.random.random(size=(3, 8))
        Orbit(pos=x, vel=v)

    with pytest.raises(ValueError):
        x = np.random.random(size=(3, 10))
        v = np.random.random(size=(3, 10))
        t = np.arange(8)
        Orbit(pos=x, vel=v, t=t)

    # TODO: always?
    # x = np.random.random(size=(3, 10))
    # v = np.random.random(size=(3, 10))
    # o = Orbit(pos=x, vel=v)
    # assert o.ndim == 3

    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.random(size=(3, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v)
    assert o.xyz.unit == u.kpc
    assert o.v_x.unit == u.km/u.s

    # TODO: don't support < 3 dim?
    # x = np.random.random(size=(2, 10))
    # v = np.random.random(size=(2, 10))
    # o = Orbit(pos=x, vel=v)
    # assert o.ndim == 2
    # assert o.hamiltonian is None

    # Check that passing in frame and potential or Hamiltonian works
    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.random(size=(3, 10))*u.km/u.s
    frame = StaticFrame(galactic)
    potential = LogarithmicPotential(v_c=1., r_h=0.14, q1=1., q2=0.9, q3=1.,
                                     units=galactic)

    o = Orbit(pos=x, vel=v, frame=frame)
    assert o.hamiltonian is None
    assert o.potential is None

    o = Orbit(pos=x, vel=v, potential=potential)
    assert o.hamiltonian is None
    assert o.frame is None

    o = Orbit(pos=x, vel=v, potential=potential, frame=frame)
    o = Orbit(pos=x, vel=v,
              hamiltonian=Hamiltonian(potential, frame=frame))
    assert isinstance(o.hamiltonian, Hamiltonian)
    assert isinstance(o.potential, LogarithmicPotential)
    assert isinstance(o.frame, StaticFrame)


def test_from_w():

    w = np.random.random(size=(6, 10))
    o = Orbit.from_w(w, galactic)
    assert o.xyz.unit == u.kpc
    assert o.v_x.unit == u.kpc/u.Myr


def test_slice():

    # simple
    x = np.random.random(size=(3, 10))
    v = np.random.random(size=(3, 10))
    o = Orbit(pos=x, vel=v)
    new_o = o[:5]
    assert new_o.shape == (5,)

    x = np.random.random(size=(3, 10))
    v = np.random.random(size=(3, 10))
    t = np.linspace(0, 10, 10)
    o = Orbit(pos=x, vel=v, t=t)
    new_o = o[:5]
    assert new_o.shape == (5,)

    # 1d slice on 3d
    x = np.random.random(size=(3, 10, 8))
    v = np.random.random(size=(3, 10, 8))
    t = np.arange(x.shape[1])
    o = Orbit(pos=x, vel=v, t=t)
    new_o = o[:5]
    assert new_o.shape == (5, 8)
    assert new_o.t.shape == (5,)

    # pick a single orbit
    new_o = o[:, 0]
    assert isinstance(new_o, Orbit)
    assert new_o.shape == (10,)
    assert new_o.t.shape == (10,)

    # pick a single time
    new_o = o[3]
    assert isinstance(new_o, PhaseSpacePosition)
    assert new_o.shape == (8,)

    # REGRESSION TEST: numpy int64 is not an int()
    new_o = o[np.int64(3)]
    assert isinstance(new_o, PhaseSpacePosition)
    assert new_o.shape == (8,)

    # 3d slice on 3d
    o = Orbit(pos=x, vel=v, t=t)
    new_o = o[:5, :4]
    assert new_o.shape == (5, 4)
    assert new_o.t.shape == (5,)

    # boolean array
    x = np.random.random(size=(3, 10))
    v = np.random.random(size=(3, 10))
    t = np.arange(x.shape[1])
    o = Orbit(pos=x, vel=v, t=t)
    ix = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).astype(bool)
    new_o = o[ix]
    assert new_o.shape == (sum(ix),)
    assert new_o.t.shape == (5,)

    # boolean array - 3D
    x = np.random.random(size=(3, 10, 4))
    v = np.random.random(size=(3, 10, 4))
    t = np.arange(x.shape[1])
    o = Orbit(pos=x, vel=v, t=t)
    ix = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).astype(bool)
    new_o = o[ix]
    assert new_o.shape == (sum(ix), x.shape[-1])
    assert new_o.t.shape == (5,)

    # integer array
    x = np.random.random(size=(3, 10))
    v = np.random.random(size=(3, 10))
    t = np.arange(x.shape[1])
    o = Orbit(pos=x, vel=v, t=t)
    ix = np.array([0, 3, 5])
    new_o = o[ix]
    assert new_o.shape == (len(ix),)
    assert new_o.t.shape == (len(ix),)


def test_reshape():
    # 1d slice on 3d
    x = np.random.random(size=(3, 10, 8))
    v = np.random.random(size=(3, 10, 8))
    t = np.arange(x.shape[1])
    o = Orbit(pos=x, vel=v, t=t)
    new_o = o.reshape((10, 4, 2))
    assert new_o.shape == (10, 4, 2)
    assert new_o.x.shape == (10, 4, 2)


def test_represent_as():

    # simple / unitless
    x = np.random.random(size=(3, 10))
    v = np.random.random(size=(3, 10))
    o = Orbit(pos=x, vel=v)
    sph = o.represent_as(SphericalRepresentation)

    assert sph.pos.distance.unit == u.one
    assert sph.vel.d_distance.unit == u.one

    # simple / with units
    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.normal(0., 100., size=(3, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v)
    sph = o.represent_as(SphericalRepresentation)
    assert sph.pos.distance.unit == u.kpc
    assert sph.vel.d_distance.unit == u.km/u.s


def test_represent_as_expected_attributes():
    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.normal(0., 100., size=(3, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v)

    new_o = o.spherical
    assert hasattr(new_o, 'distance')
    assert hasattr(new_o, 'lat')
    assert hasattr(new_o, 'lon')
    assert hasattr(new_o, 'radial_velocity')
    assert hasattr(new_o, 'pm_lat')
    assert hasattr(new_o, 'pm_lon')
    assert new_o.norbits == o.norbits

    new_o = o.represent_as(SphericalRepresentation, SphericalCosLatDifferential)
    assert hasattr(new_o, 'distance')
    assert hasattr(new_o, 'lat')
    assert hasattr(new_o, 'lon')
    assert hasattr(new_o, 'radial_velocity')
    assert hasattr(new_o, 'pm_lat')
    assert hasattr(new_o, 'pm_lon_coslat')

    new_o = o.physicsspherical
    assert hasattr(new_o, 'r')
    assert hasattr(new_o, 'phi')
    assert hasattr(new_o, 'theta')
    assert hasattr(new_o, 'radial_velocity')
    assert hasattr(new_o, 'pm_theta')
    assert hasattr(new_o, 'pm_phi')
    assert new_o.norbits == o.norbits

    new_o = o.cylindrical
    assert hasattr(new_o, 'rho')
    assert hasattr(new_o, 'phi')
    assert hasattr(new_o, 'z')
    assert hasattr(new_o, 'v_rho')
    assert hasattr(new_o, 'pm_phi')
    assert hasattr(new_o, 'v_z')
    assert new_o.norbits == o.norbits

    new_o = new_o.cartesian
    assert hasattr(new_o, 'x')
    assert hasattr(new_o, 'y')
    assert hasattr(new_o, 'z')
    assert hasattr(new_o, 'xyz')
    assert hasattr(new_o, 'v_x')
    assert hasattr(new_o, 'v_y')
    assert hasattr(new_o, 'v_z')
    assert hasattr(new_o, 'v_xyz')

    # Check that this works with the NDCartesian classes too
    x = np.random.random(size=(2, 10))*u.kpc
    v = np.random.normal(0., 100., size=(2, 10))*u.km/u.s
    new_o = Orbit(pos=x, vel=v)

    assert hasattr(new_o, 'x1')
    assert hasattr(new_o, 'x2')
    assert hasattr(new_o, 'xyz')
    assert hasattr(new_o, 'v_x1')
    assert hasattr(new_o, 'v_x2')
    assert hasattr(new_o, 'v_xyz')


def test_to_coord_frame():
    # simple / unitless
    x = np.random.random(size=(3, 10))
    v = np.random.random(size=(3, 10))
    o = Orbit(pos=x, vel=v)

    with pytest.raises(u.UnitConversionError):
        o.to_coord_frame(Galactic())

    # simple / with units
    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.normal(0., 100., size=(3, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v)
    coo = o.to_coord_frame(Galactic())
    assert coo.name == 'galactic'

    # simple / with units and time
    x = np.random.random(size=(3, 128, 10))*u.kpc
    v = np.random.normal(0., 100., size=(3, 128, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v)
    coo = o.to_coord_frame(Galactic())
    assert coo.name == 'galactic'


def test_w():
    # simple / unitless
    x = np.random.random(size=(3, 10))
    v = np.random.random(size=(3, 10))
    o = Orbit(pos=x, vel=v)
    w = o.w()
    assert w.shape == (6, 10)

    # simple / with units
    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.normal(0., 100., size=(3, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v)
    with pytest.raises(ValueError):
        o.w()
    w = o.w(units=galactic)
    assert np.allclose(x.value, w[:3, :])
    assert np.allclose(v.value, (w[3:, :]*u.kpc/u.Myr).to(u.km/u.s).value)

    # simple / with units and potential
    p = HernquistPotential(units=galactic, m=1E11, c=0.25)
    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.normal(0., 100., size=(3, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v, potential=p, frame=StaticFrame(galactic))
    w = o.w()
    assert np.allclose(x.value, w[:3, :])
    assert np.allclose(v.value, (w[3:, :]*u.kpc/u.Myr).to(u.km/u.s).value)

    w = o.w(units=solarsystem)
    assert np.allclose(x.value, (w[:3, :]*u.au).to(u.kpc).value)
    assert np.allclose(v.value, (w[3:, :]*u.au/u.yr).to(u.km/u.s).value)


def test_energy():
    # with units
    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.normal(0., 100., size=(3, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v)
    KE = o.kinetic_energy()
    assert KE.unit == (o.v_x.unit)**2
    assert KE.shape == o.pos.shape

    # with units and potential
    p = HernquistPotential(units=galactic, m=1E11, c=0.25)
    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.normal(0., 100., size=(3, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v, potential=p, frame=StaticFrame(galactic))
    o.potential_energy()
    o.energy()


def test_angular_momentum():
    # with units
    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.normal(0., 100., size=(3, 10))*u.km/u.s
    o = Orbit(pos=x, vel=v)
    L = o.angular_momentum()
    assert L.unit == (o.v_x.unit*o.x.unit)
    assert L.shape == ((3,) + o.shape)


def test_eccentricity():
    pot = KeplerPotential(m=1., units=solarsystem)
    w0 = PhaseSpacePosition(pos=[1, 0, 0.]*u.au,
                            vel=[0., 2*np.pi, 0.]*u.au/u.yr)
    ham = Hamiltonian(pot)
    w = ham.integrate_orbit(w0, dt=0.01, n_steps=10000, Integrator=DOPRI853Integrator)
    e = w.eccentricity()
    assert np.abs(e) < 1E-3


def test_apocenter_pericenter_period():
    pot = KeplerPotential(m=1., units=solarsystem)
    w0 = PhaseSpacePosition(pos=[1, 0, 0.]*u.au,
                            vel=[0., 1.5*np.pi, 0.]*u.au/u.yr)

    ham = Hamiltonian(pot)
    w = ham.integrate_orbit(w0, dt=0.01, n_steps=10000,
                            Integrator=DOPRI853Integrator)

    apo = w.apocenter()
    per = w.pericenter()
    zmax = w.zmax()
    assert apo.shape == ()
    assert per.shape == ()
    assert zmax.shape == ()

    assert apo.unit == u.au
    assert per.unit == u.au
    assert zmax.unit == u.au
    assert apo > per

    # see if they're where we expect
    E = np.mean(w.energy()).decompose(pot.units).value
    L = np.mean(np.sqrt(np.sum(w.angular_momentum()**2, axis=0))).decompose(pot.units).value

    def func(r):
        val = 2*(E-pot.energy([r, 0, 0]).value[0]) - L**2/r**2
        return val

    pred_apo = so.brentq(func, 0.9, 1.0)
    pred_per = so.brentq(func, 0.3, 0.5)

    assert np.allclose(apo.value, pred_apo, rtol=1E-2)
    assert np.allclose(per.value, pred_per, rtol=1E-2)

    # Return all peris, apos
    apos = w.apocenter(func=None)
    pers = w.pericenter(func=None)
    zmax = w.zmax(func=None)
    T = w.estimate_period()  # noqa

    dapo = np.std(apos) / np.mean(apos)
    assert (dapo > 0) and np.allclose(dapo, 0., atol=1E-4)

    dper = np.std(pers) / np.mean(pers)
    assert (dper > 0) and np.allclose(dper, 0., atol=1E-4)

    # Now try for expected behavior when multiple orbits are integrated:
    w0 = PhaseSpacePosition(pos=([[1, 0, 0.], [1.1, 0, 0]]*u.au).T,
                            vel=([[0., 1.5*np.pi, 0.],
                                  [0., 1.5*np.pi, 0.]]*u.au/u.yr).T)

    w = ham.integrate_orbit(w0, dt=0.01, n_steps=10000)

    per = w.pericenter(approximate=True)
    apo = w.apocenter(approximate=True)
    zmax = w.zmax(approximate=True)
    ecc = w.eccentricity(approximate=True)  # noqa


def test_estimate_period():
    ntimes = 16384
    for true_T_R in [1., 2., 4.123]:
        t = np.linspace(0, 10., ntimes)
        R = 0.25*np.sin(2*np.pi/true_T_R * t) + 1.
        phi = (2*np.pi * t) % (2*np.pi)

        pos = np.zeros((3, ntimes))
        pos[0] = R*np.cos(phi)
        pos[1] = R*np.sin(phi)
        vel = np.zeros_like(pos)

        orb = Orbit(pos*u.kpc, vel*u.kpc/u.Myr, t=t*u.Gyr)
        T = orb.estimate_period()
        assert np.allclose(T.value, true_T_R, rtol=1E-3)


def test_estimate_period_regression():
    pot = KeplerPotential(m=1., units=solarsystem)
    w0 = PhaseSpacePosition(pos=[1, 0, 0.]*u.au,
                            vel=[0., 1.5*np.pi, 0.]*u.au/u.yr)
    w0 = combine((w0, w0, w0))

    ham = Hamiltonian(pot)
    w = ham.integrate_orbit(w0, dt=0.01, n_steps=10000,
                            Integrator=DOPRI853Integrator)
    T = w.estimate_period()
    print(T)


def make_known_orbits(tmpdir, xs, vxs, potential, names):
    # See Binney & Tremaine (2008) Figure 3.8 and 3.9
    E = -0.337
    y = 0.

    ws = []
    for x, vx, name in zip(xs, vxs, names):
        vy = np.sqrt(2*(E - potential.energy([x, y, 0.]).value))[0]
        w = [x, y, 0., vx, vy, 0.]
        ws.append(w)
    ws = np.array(ws).T

    ham = Hamiltonian(potential)
    orbit = ham.integrate_orbit(ws, dt=0.05, n_steps=10000)

    return orbit


def test_circulation(tmpdir):

    potential = LogarithmicPotential(v_c=1., r_h=0.14, q1=1., q2=0.9, q3=1.,
                                     units=galactic)

    # individual
    ws = make_known_orbits(tmpdir, [0.5, 0], [0., 1.5],
                           potential, ["loop", "box"])

    w1 = ws[:, 0]
    circ = w1.circulation()
    assert circ.shape == (3,)
    assert circ.sum() == 1

    w2 = ws[:, 1]
    circ = w2.circulation()
    assert circ.shape == (3,)
    assert circ.sum() == 0

    # try also for both, together
    circ = ws.circulation()
    assert circ.shape == (3, 2)
    assert np.allclose(circ.sum(axis=0), [1, 0])


def test_align_circulation():

    t = np.linspace(0, 100, 1024)
    w = np.zeros((6, 1024, 4))

    # loop around x axis
    w[1, :, 0] = np.cos(t)
    w[2, :, 0] = np.sin(t)
    w[4, :, 0] = -np.sin(t)
    w[5, :, 0] = np.cos(t)

    # loop around y axis
    w[0, :, 1] = -np.cos(t)
    w[2, :, 1] = np.sin(t)
    w[3, :, 1] = np.sin(t)
    w[5, :, 1] = np.cos(t)

    # loop around z axis
    w[0, :, 2] = np.cos(t)
    w[1, :, 2] = np.sin(t)
    w[3, :, 2] = -np.sin(t)
    w[4, :, 2] = np.cos(t)

    # box
    w[0, :, 3] = np.cos(t)
    w[1, :, 3] = -np.cos(0.5*t)
    w[2, :, 3] = np.cos(0.25*t)
    w[3, :, 3] = -np.sin(t)
    w[4, :, 3] = 0.5*np.sin(0.5*t)
    w[5, :, 3] = -0.25*np.sin(0.25*t)

    # First, individually
    for i in range(w.shape[2]):
        orb = Orbit.from_w(w[..., i], units=galactic)
        new_orb = orb.align_circulation_with_z()
        circ = new_orb.circulation()

        if i == 3:
            assert np.sum(circ) == 0
        else:
            assert circ[2] == 1.

    # all together now
    orb = Orbit.from_w(w, units=galactic)
    circ = orb.circulation()
    assert circ.shape == (3, 4)

    new_orb = orb.align_circulation_with_z()
    new_circ = new_orb.circulation()
    assert np.all(new_circ[2, :3] == 1.)
    assert np.all(new_circ[:, 3] == 0.)


def test_frame_transform():
    static = StaticFrame(galactic)
    rotating = ConstantRotatingFrame(Omega=[0.53, 1.241, 0.9394]*u.rad/u.Myr, units=galactic)

    x = np.random.random(size=(3, 10))*u.kpc
    v = np.random.random(size=(3, 10))*u.km/u.s
    t = np.linspace(0, 1, 10)*u.Myr

    # no frame specified at init
    o = Orbit(pos=x, vel=v, t=t)
    with pytest.raises(ValueError):
        o.to_frame(rotating)

    o.to_frame(rotating, current_frame=static, t=o.t)
    o.to_frame(rotating, current_frame=static)

    # frame specified at init
    o = Orbit(pos=x, vel=v, t=t,
              frame=static,
              potential=HernquistPotential(m=1E10, c=0.5, units=galactic))
    o.to_frame(rotating)
    o.to_frame(rotating, t=o.t)


_x = ([[1, 2, 3.], [1, 2, 3.]]*u.kpc).T
_v = ([[1, 2, 3.], [1, 2, 3.]]*u.km/u.s).T
@pytest.mark.parametrize('obj', [  # noqa
    Orbit(_x, _v),
    Orbit(_x, _v, t=[5, 99]*u.Myr),
    Orbit(_x, _v, t=[5, 99]*u.Myr,
          frame=StaticFrame(galactic)),
    Orbit(_x, _v, t=[5, 99]*u.Myr,
          frame=StaticFrame(galactic),
          potential=HernquistPotential(m=1E10, c=0.5, units=galactic)),
])
@pytest.mark.skipif(not HAS_H5PY, reason='h5py required for this test')
def test_io(tmpdir, obj):
    import h5py

    filename = str(tmpdir.join('thing.hdf5'))
    with h5py.File(filename, 'w') as f:
        obj.to_hdf5(f)

    obj2 = Orbit.from_hdf5(filename)
    assert u.allclose(obj.xyz, obj2.xyz)
    assert u.allclose(obj.v_xyz, obj2.v_xyz)
    if obj.t:
        assert u.allclose(obj.t, obj2.t)

    assert obj.frame == obj2.frame
    assert obj.potential == obj2.potential


@pytest.mark.parametrize('obj', [
    Orbit(_x, _v),
    Orbit(_x, _v, t=[5, 99]*u.Myr),
    Orbit(_x, _v, t=[5, 99]*u.Myr,
          frame=StaticFrame(galactic)),
    Orbit(_x, _v, t=[5, 99]*u.Myr,
          frame=StaticFrame(galactic),
          potential=HernquistPotential(m=1E10, c=0.5, units=galactic)),
])
@pytest.mark.skipif(not HAS_GALPY,
                    reason="requires galpy to run this test")
def test_orbit_to_galpy(obj):
    o1 = obj.to_galpy_orbit()  # noqa
    o2 = obj.to_galpy_orbit(ro=8*u.kpc)  # noqa
    o3 = obj.to_galpy_orbit(vo=220*u.km/u.s)  # noqa
    o4 = obj.to_galpy_orbit(ro=8*u.kpc, vo=220*u.km/u.s)  # noqa


@pytest.mark.skipif(not HAS_GALPY,
                    reason="requires galpy to run this test")
def test_orbit_from_galpy():
    import galpy.orbit as galpy_o
    import galpy.potential as galpy_p
    mp = galpy_p.MiyamotoNagaiPotential(a=0.5, b=0.0375, amp=1., normalize=1.)
    galpy_orbit = galpy_o.Orbit([1., 0.1, 1.1, 0., 0.1, 1.])
    ts = np.linspace(0, 100, 10000)
    galpy_orbit.integrate(ts, mp, method='odeint')
    gala_orbit = Orbit.from_galpy_orbit(galpy_orbit)

    assert len(gala_orbit.t) == len(ts)
