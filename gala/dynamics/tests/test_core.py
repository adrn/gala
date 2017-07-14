# coding: utf-8

""" Test core dynamics.  """

from __future__ import division, print_function

# Standard library
import warnings

# Third-party
import astropy.units as u
from astropy.coordinates import (Galactic, CartesianRepresentation,
                                 SphericalRepresentation, CartesianDifferential,
                                 SphericalDifferential,
                                 SphericalCosLatDifferential)
from astropy.tests.helper import quantity_allclose
import numpy as np
import pytest

# Project
from ..core import PhaseSpacePosition
from ...potential import Hamiltonian, HernquistPotential
from ...potential.frame import StaticFrame, ConstantRotatingFrame
from ...units import galactic, solarsystem

def test_initialize():

    with pytest.raises(ValueError):
        x = np.random.random(size=(3,10))
        v = np.random.random(size=(3,8))
        PhaseSpacePosition(pos=x, vel=v)

    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    assert o.shape == (10,)

    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.random(size=(3,10))*u.km/u.s
    o = PhaseSpacePosition(pos=x, vel=v)
    assert o.xyz.unit == u.kpc
    assert o.v_x.unit == u.km/u.s

    # Not 3D
    x = np.random.random(size=(2,10))
    v = np.random.random(size=(2,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    assert o.ndim == 2

    o = PhaseSpacePosition(pos=x, vel=v, frame=StaticFrame(galactic))
    assert o.ndim == 2
    assert o.frame is not None
    assert isinstance(o.frame, StaticFrame)

    x = np.random.random(size=(4,10))
    v = np.random.random(size=(4,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    assert o.ndim == 4

    # back to 3D
    pos = CartesianRepresentation(np.random.random(size=(3,10))*u.one)
    vel = CartesianDifferential(np.random.random(size=(3,10))*u.one)
    o = PhaseSpacePosition(pos=pos, vel=vel)
    assert hasattr(o, 'x')
    assert hasattr(o, 'y')
    assert hasattr(o, 'z')
    assert hasattr(o, 'v_x')
    assert hasattr(o, 'v_y')
    assert hasattr(o, 'v_z')

    # passing a representation with a differential attached
    pos = CartesianRepresentation(np.random.random(size=(3,10))*u.kpc)
    vel = CartesianDifferential(np.random.random(size=(3,10))*u.km/u.s)
    o = PhaseSpacePosition(pos.with_differentials({'s': vel}))
    assert hasattr(o, 'x')
    assert hasattr(o, 'y')
    assert hasattr(o, 'z')
    assert hasattr(o, 'v_x')
    assert hasattr(o, 'v_y')
    assert hasattr(o, 'v_z')

    o = o.represent_as(SphericalRepresentation)
    assert hasattr(o, 'distance')
    assert hasattr(o, 'lat')
    assert hasattr(o, 'lon')
    assert hasattr(o, 'radial_velocity')
    assert hasattr(o, 'pm_lon')
    assert hasattr(o, 'pm_lat')

    with pytest.raises(TypeError):
        o = PhaseSpacePosition(pos=x, vel=v, frame="blah blah blah")

    # check that old class raises deprecation warning
    from ..core import CartesianPhaseSpacePosition
    warnings.simplefilter('always')
    with pytest.warns(DeprecationWarning):
        o = CartesianPhaseSpacePosition(pos=x, vel=v)

def test_from_w():

    w = np.random.random(size=(6,10))
    o = PhaseSpacePosition.from_w(w, galactic)
    assert o.x.unit == u.kpc
    assert o.v_x.unit == u.kpc/u.Myr
    assert o.shape == (10,)

def test_slice():

    # simple
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    new_o = o[:5]
    assert new_o.shape == (5,)

    # 1d slice on 3d
    x = np.random.random(size=(3,10,8))
    v = np.random.random(size=(3,10,8))
    o = PhaseSpacePosition(pos=x, vel=v)
    new_o = o[:5]
    assert new_o.shape == (5,8)

    # 3d slice on 3d
    o = PhaseSpacePosition(pos=x, vel=v)
    new_o = o[:5,:4]
    assert new_o.shape == (5,4)

    # boolean array
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    ix = np.array([0,0,0,0,0,1,1,1,1,1]).astype(bool)
    new_o = o[ix]
    assert new_o.shape == (sum(ix),)

    # integer array
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    ix = np.array([0,3,5])
    new_o = o[ix]
    assert new_o.shape == (len(ix),)

# ------------------------------------------------------------------------
# Convert from Cartesian to other representations
# ------------------------------------------------------------------------
def test_represent_as():

    # simple / unitless
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    new_o = o.represent_as(SphericalRepresentation)
    o.spherical
    o.cylindrical
    o.cartesian

    assert new_o.pos.distance.unit == u.one
    assert new_o.vel.d_distance.unit == u.one

    # simple / with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = PhaseSpacePosition(pos=x, vel=v)
    sph = o.represent_as(SphericalRepresentation)
    assert sph.pos.distance.unit == u.kpc

    sph2 = o.represent_as('spherical')
    for c in sph.pos.components:
        assert quantity_allclose(getattr(sph.pos, c),
                                 getattr(sph2.pos, c),
                                 rtol=1E-12)

    # doesn't work for 2D
    x = np.random.random(size=(2,10))
    v = np.random.random(size=(2,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    with pytest.raises(ValueError):
        o.represent_as(SphericalRepresentation)

def test_represent_as_expected_attributes():
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = PhaseSpacePosition(pos=x, vel=v)

    new_o = o.spherical
    assert hasattr(new_o, 'distance')
    assert hasattr(new_o, 'lat')
    assert hasattr(new_o, 'lon')
    assert hasattr(new_o, 'radial_velocity')
    assert hasattr(new_o, 'pm_lat')
    assert hasattr(new_o, 'pm_lon')

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

    new_o = o.cylindrical
    assert hasattr(new_o, 'rho')
    assert hasattr(new_o, 'phi')
    assert hasattr(new_o, 'z')
    assert hasattr(new_o, 'v_rho')
    assert hasattr(new_o, 'pm_phi')
    assert hasattr(new_o, 'v_z')

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
    x = np.random.random(size=(2,10))*u.kpc
    v = np.random.normal(0.,100.,size=(2,10))*u.km/u.s
    new_o = PhaseSpacePosition(pos=x, vel=v)

    assert hasattr(new_o, 'x1')
    assert hasattr(new_o, 'x2')
    assert hasattr(new_o, 'xyz')
    assert hasattr(new_o, 'v_x1')
    assert hasattr(new_o, 'v_x2')
    assert hasattr(new_o, 'v_xyz')

def test_to_coord_frame():
    # simple / unitless
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = PhaseSpacePosition(pos=x, vel=v)

    with pytest.raises(u.UnitConversionError):
        o.to_coord_frame(Galactic)

    # simple / with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = PhaseSpacePosition(pos=x, vel=v)
    coo = o.to_coord_frame(Galactic)
    assert coo.name == 'galactic'

    warnings.simplefilter('always')
    with pytest.warns(DeprecationWarning):
        o.to_frame(Galactic)

    # doesn't work for 2D
    x = np.random.random(size=(2,10))*u.kpc
    v = np.random.normal(0.,100.,size=(2,10))*u.km/u.s
    o = PhaseSpacePosition(pos=x, vel=v)
    with pytest.raises(ValueError):
        o.to_coord_frame(Galactic)

def test_w():
    # simple / unitless
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    w = o.w()
    assert w.shape == (6,10)

    x = np.random.random(size=3)
    v = np.random.random(size=3)
    o = PhaseSpacePosition(pos=x, vel=v)
    w = o.w()
    assert w.shape == (6,1)

    # simple / unitless, 2D
    x = np.random.random(size=(2,10))
    v = np.random.random(size=(2,10))
    o = PhaseSpacePosition(pos=x, vel=v)
    w = o.w()
    assert w.shape == (4,10)

    x = np.random.random(size=2)
    v = np.random.random(size=2)
    o = PhaseSpacePosition(pos=x, vel=v)
    w = o.w()
    assert w.shape == (4,1)

    # simple / with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = PhaseSpacePosition(pos=x, vel=v)
    with pytest.raises(ValueError):
        o.w()
    w = o.w(units=galactic)
    assert np.allclose(x.value, w[:3])
    assert np.allclose(v.value, (w[3:]*u.kpc/u.Myr).to(u.km/u.s).value)

    # simple / with units and potential
    p = HernquistPotential(units=galactic, m=1E11, c=0.25)
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = PhaseSpacePosition(pos=x, vel=v)
    w = o.w(p.units)
    assert np.allclose(x.value, w[:3])
    assert np.allclose(v.value, (w[3:]*u.kpc/u.Myr).to(u.km/u.s).value)

    w = o.w(units=solarsystem)
    assert np.allclose(x.value, (w[:3]*u.au).to(u.kpc).value)
    assert np.allclose(v.value, (w[3:]*u.au/u.yr).to(u.km/u.s).value)

# ------------------------------------------------------------------------
# Computed dynamical quantities
# ------------------------------------------------------------------------
def test_energy():
    # with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = PhaseSpacePosition(pos=x, vel=v)
    KE = o.kinetic_energy()
    assert KE.unit == (o.v_x.unit)**2
    assert KE.shape == o.shape

    # with units and potential
    p = HernquistPotential(units=galactic, m=1E11, c=0.25)
    H = Hamiltonian(p)
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = PhaseSpacePosition(pos=x, vel=v)
    PE = o.potential_energy(p)
    E = o.energy(H)

    warnings.simplefilter('always')
    with pytest.warns(DeprecationWarning):
        o.energy(p)

def test_angular_momentum():

    w = PhaseSpacePosition([1.,0.,0.], [0.,0.,1.])
    assert quantity_allclose(np.squeeze(w.angular_momentum()), [0.,-1,0]*u.one)

    w = PhaseSpacePosition([1.,0.,0.], [0.,1.,0.])
    assert quantity_allclose(np.squeeze(w.angular_momentum()), [0.,0, 1]*u.one)

    w = PhaseSpacePosition([0.,1.,0.],[0.,0.,1.])
    assert quantity_allclose(np.squeeze(w.angular_momentum()), [1., 0, 0]*u.one)

    w = PhaseSpacePosition([1.,0,0]*u.kpc, [0.,200.,0]*u.pc/u.Myr)
    assert quantity_allclose(np.squeeze(w.angular_momentum()), [0,0,0.2]*u.kpc**2/u.Myr)

    # multiple - known
    q = np.array([[1.,0.,0.],[1.,0.,0.],[0,1.,0.]]).T
    p = np.array([[0,0,1.],[0,1.,0.],[0,0,1]]).T
    L = PhaseSpacePosition(q, p).angular_momentum()
    true_L = np.array([[0., -1, 0],[0., 0, 1],[1., 0, 0]]).T * u.one
    assert L.shape == (3,3)
    assert quantity_allclose(L, true_L)

    # multiple - random
    q = np.random.uniform(size=(3,128))
    p = np.random.uniform(size=(3,128))
    L = PhaseSpacePosition(q, p).angular_momentum()
    assert L.shape == (3,128)

def test_frame_transform():
    static = StaticFrame(galactic)
    rotating = ConstantRotatingFrame(Omega=[0.53,1.241,0.9394]*u.rad/u.Myr,
                                     units=galactic)

    x = np.array([[10.,-0.2,0.3],[-0.232,8.1,0.1934]]).T * u.kpc
    v = np.array([[0.0034,0.2,0.0014],[0.0001,0.002532,-0.2]]).T * u.kpc/u.Myr

    # no frame specified at init
    psp = PhaseSpacePosition(pos=x, vel=v)
    with pytest.raises(ValueError):
        psp.to_frame(rotating)

    psp.to_frame(rotating, current_frame=static, t=0.4*u.Myr)

    # frame specified at init
    psp = PhaseSpacePosition(pos=x, vel=v, frame=static)
    psp.to_frame(rotating, t=0.4*u.Myr)

@pytest.mark.parametrize('obj', [
    PhaseSpacePosition([1,2,3.]*u.kpc, [1,2,3.]*u.km/u.s),
    PhaseSpacePosition([1,2,3.]*u.kpc, [1,2,3.]*u.km/u.s,
                       StaticFrame(galactic)),
    PhaseSpacePosition([1,2,3.]*u.kpc, [1,2,3.]*u.km/u.s,
                       ConstantRotatingFrame([1.,0,0]*u.rad/u.Myr, galactic)),
])
def test_io(tmpdir, obj):
    import h5py

    filename = str(tmpdir.join('thing.hdf5'))
    with h5py.File(filename, 'w') as f:
        obj.to_hdf5(f)

    obj2 = PhaseSpacePosition.from_hdf5(filename)
    assert quantity_allclose(obj.xyz, obj2.xyz)
    assert quantity_allclose(obj.v_xyz, obj2.v_xyz)
    assert obj.frame == obj2.frame
