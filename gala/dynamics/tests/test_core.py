# coding: utf-8

""" Test core dynamics.  """

from __future__ import division, print_function

# Standard library
import warnings

# Third-party
import astropy.units as u
from astropy.coordinates import SphericalRepresentation, Galactic
from astropy.tests.helper import quantity_allclose
import numpy as np
import pytest

# Project
from ..core import *
from ...potential import Hamiltonian, HernquistPotential
from ...potential.frame import StaticFrame, ConstantRotatingFrame
from ...units import galactic, solarsystem

def test_initialize():

    with pytest.raises(ValueError):
        x = np.random.random(size=(3,10))
        v = np.random.random(size=(3,8))
        CartesianPhaseSpacePosition(pos=x, vel=v)

    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    assert o.ndim == 3

    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.random(size=(3,10))*u.km/u.s
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    assert o.pos.unit == u.kpc
    assert o.vel.unit == u.km/u.s

    x = np.random.random(size=(2,10))
    v = np.random.random(size=(2,10))
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    assert o.ndim == 2

    o = CartesianPhaseSpacePosition(pos=x, vel=v, frame=StaticFrame(galactic))
    assert o.ndim == 2
    assert o.frame is not None
    assert isinstance(o.frame, StaticFrame)

    with pytest.raises(TypeError):
        o = CartesianPhaseSpacePosition(pos=x, vel=v, frame="blah blah blah")

def test_from_w():

    w = np.random.random(size=(6,10))
    o = CartesianPhaseSpacePosition.from_w(w, galactic)
    assert o.pos.unit == u.kpc
    assert o.vel.unit == u.kpc/u.Myr

def test_slice():

    # simple
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    new_o = o[:5]
    assert new_o.pos.shape == (3,5)
    assert new_o.vel.shape == (3,5)

    # 1d slice on 3d
    x = np.random.random(size=(3,100,8))
    v = np.random.random(size=(3,100,8))
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    new_o = o[:5]
    assert new_o.pos.shape == (3,5,8)
    assert new_o.vel.shape == (3,5,8)

    # 3d slice on 3d
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    new_o = o[:5,:4]
    assert new_o.pos.shape == (3,5,4)
    assert new_o.vel.shape == (3,5,4)

    # boolean array
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    ix = np.array([0,0,0,0,0,1,1,1,1,1]).astype(bool)
    new_o = o[ix]
    assert new_o.shape == (sum(ix),)

    # integer array
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
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
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    sph_pos, sph_vel = o.represent_as(SphericalRepresentation)
    o.spherical
    o.cylindrical
    o.cartesian

    assert sph_pos.distance.unit == u.dimensionless_unscaled
    assert sph_vel.unit == u.dimensionless_unscaled

    # simple / with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    sph_pos, sph_vel = o.represent_as(SphericalRepresentation)
    assert sph_pos.distance.unit == u.kpc

    # doesn't work for 2D
    x = np.random.random(size=(2,10))
    v = np.random.random(size=(2,10))
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    with pytest.raises(ValueError):
        o.represent_as(SphericalRepresentation)

def test_to_coord_frame():
    # simple / unitless
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianPhaseSpacePosition(pos=x, vel=v)

    with pytest.raises(u.UnitConversionError):
        o.to_coord_frame(Galactic)

    # simple / with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    coo,vel = o.to_coord_frame(Galactic)
    assert coo.name == 'galactic'

    warnings.simplefilter('always')
    with pytest.warns(DeprecationWarning):
        o.to_frame(Galactic)

    # doesn't work for 2D
    x = np.random.random(size=(2,10))*u.kpc
    v = np.random.normal(0.,100.,size=(2,10))*u.km/u.s
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    with pytest.raises(ValueError):
        o.to_coord_frame(Galactic)

def test_w():
    # simple / unitless
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    w = o.w()
    assert w.shape == (6,10)

    # simple / with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    with pytest.raises(ValueError):
        o.w()
    w = o.w(units=galactic)
    assert np.allclose(x.value, w[:3])
    assert np.allclose(v.value, (w[3:]*u.kpc/u.Myr).to(u.km/u.s).value)

    # simple / with units and potential
    p = HernquistPotential(units=galactic, m=1E11, c=0.25)
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
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
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    KE = o.kinetic_energy()
    assert KE.unit == (o.vel.unit)**2
    assert KE.shape == o.pos.shape[1:]

    # with units and potential
    p = HernquistPotential(units=galactic, m=1E11, c=0.25)
    H = Hamiltonian(p)
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = CartesianPhaseSpacePosition(pos=x, vel=v)
    PE = o.potential_energy(p)
    E = o.energy(H)

    warnings.simplefilter('always')
    with pytest.warns(DeprecationWarning):
        o.energy(p)

def test_angular_momentum():

    w = CartesianPhaseSpacePosition([1.,0.,0.], [0.,0.,1.])
    assert np.allclose(np.squeeze(w.angular_momentum()), [0., -1, 0])

    w = CartesianPhaseSpacePosition([1.,0.,0.], [0.,1.,0.])
    assert np.allclose(np.squeeze(w.angular_momentum()), [0., 0, 1])

    w = CartesianPhaseSpacePosition([0.,1.,0.],[0.,0.,1.])
    assert np.allclose(np.squeeze(w.angular_momentum()), [1., 0, 0])

    w = CartesianPhaseSpacePosition([1.,0,0]*u.kpc, [0.,200.,0]*u.pc/u.Myr)
    assert quantity_allclose(np.squeeze(w.angular_momentum()), [0,0,0.2]*u.kpc**2/u.Myr)

    # multiple - known
    q = np.array([[1.,0.,0.],[1.,0.,0.],[0,1.,0.]]).T
    p = np.array([[0,0,1.],[0,1.,0.],[0,0,1]]).T
    L = CartesianPhaseSpacePosition(q, p).angular_momentum()
    true_L = np.array([[0., -1, 0],[0., 0, 1],[1., 0, 0]]).T
    assert L.shape == (3,3)
    assert np.allclose(L, true_L)

    # multiple - random
    q = np.random.uniform(size=(3,128))
    p = np.random.uniform(size=(3,128))
    L = CartesianPhaseSpacePosition(q, p).angular_momentum()
    assert L.shape == (3,128)

def test_combine():

    o1 = CartesianPhaseSpacePosition.from_w(np.random.random(size=6), units=galactic)
    o2 = CartesianPhaseSpacePosition.from_w(np.random.random(size=6), units=galactic)
    o = combine((o1, o2))
    assert o.pos.shape == (3,2)
    assert o.vel.shape == (3,2)

    o1 = CartesianPhaseSpacePosition.from_w(np.random.random(size=6), units=galactic)
    o2 = CartesianPhaseSpacePosition.from_w(np.random.random(size=(6,10)), units=galactic)
    o = combine((o1, o2))
    assert o.pos.shape == (3,11)
    assert o.vel.shape == (3,11)

    o1 = CartesianPhaseSpacePosition.from_w(np.random.random(size=6), units=galactic)
    o2 = CartesianPhaseSpacePosition.from_w(np.random.random(size=6), units=solarsystem)
    o = combine((o1, o2))
    assert o.pos.unit == galactic['length']
    assert o.vel.unit == galactic['length']/galactic['time']

    o1 = CartesianPhaseSpacePosition.from_w(np.random.random(size=6), units=galactic)
    x = np.random.random(size=(2,10))*u.kpc
    v = np.random.normal(0.,100.,size=(2,10))*u.km/u.s
    o2 = CartesianPhaseSpacePosition(pos=x, vel=v)
    with pytest.raises(ValueError):
        o = combine((o1, o2))

def test_frame_transform():
    static = StaticFrame(galactic)
    rotating = ConstantRotatingFrame(Omega=[0.53,1.241,0.9394]*u.rad/u.Myr, units=galactic)

    x = np.array([[10.,-0.2,0.3],[-0.232,8.1,0.1934]]).T * u.kpc
    v = np.array([[0.0034,0.2,0.0014],[0.0001,0.002532,-0.2]]).T * u.kpc/u.Myr

    # no frame specified at init
    psp = CartesianPhaseSpacePosition(pos=x, vel=v)
    with pytest.raises(ValueError):
        psp.to_frame(rotating)

    psp.to_frame(rotating, current_frame=static, t=0.4*u.Myr)

    # frame specified at init
    psp = CartesianPhaseSpacePosition(pos=x, vel=v, frame=static)
    psp.to_frame(rotating, t=0.4*u.Myr)
