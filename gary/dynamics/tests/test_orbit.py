# coding: utf-8

""" Test ...  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.coordinates import SphericalRepresentation, Galactic
import astropy.units as u
import numpy as np
import pytest

# Project
from ..orbit import *
from ...potential import HernquistPotential
from ...units import galactic, solarsystem

def test_initialize():

    with pytest.raises(ValueError):
        x = np.random.random(size=(3,10))
        v = np.random.random(size=(3,8))
        CartesianOrbit(pos=x, vel=v)

    with pytest.raises(ValueError):
        x = np.random.random(size=(3,10))
        v = np.random.random(size=(3,10))
        t = np.arange(8)
        CartesianOrbit(pos=x, vel=v, t=t)

    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianOrbit(pos=x, vel=v)
    assert o.ndim == 3

    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.random(size=(3,10))*u.km/u.s
    o = CartesianOrbit(pos=x, vel=v)
    assert o.pos.unit == u.kpc
    assert o.vel.unit == u.km/u.s

    x = np.random.random(size=(2,10))
    v = np.random.random(size=(2,10))
    o = CartesianOrbit(pos=x, vel=v)
    assert o.ndim == 2

def test_slice():

    # simple
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianOrbit(pos=x, vel=v)
    new_o = o[:5]
    assert new_o.pos.shape == (3,5)

    # 1d slice on 3d
    x = np.random.random(size=(3,100,8))
    v = np.random.random(size=(3,100,8))
    t = np.arange(x.shape[1])
    o = CartesianOrbit(pos=x, vel=v, t=t)
    new_o = o[:5]
    assert new_o.pos.shape == (3,5,8)
    assert new_o.t.shape == (5,)

    # 3d slice on 3d
    o = CartesianOrbit(pos=x, vel=v, t=t)
    new_o = o[:5,:4]
    assert new_o.pos.shape == (3,5,4)
    assert new_o.t.shape == (5,)

def test_represent_as():

    # simple / unitless
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianOrbit(pos=x, vel=v)
    sph_pos, sph_vel = o.represent_as(SphericalRepresentation)

    assert sph_pos.distance.unit == u.dimensionless_unscaled
    assert sph_vel.unit == u.dimensionless_unscaled

    # simple / with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = CartesianOrbit(pos=x, vel=v)
    sph_pos, sph_vel = o.represent_as(SphericalRepresentation)
    assert sph_pos.distance.unit == u.kpc

def test_to_frame():
    # simple / unitless
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianOrbit(pos=x, vel=v)

    with pytest.raises(u.UnitConversionError):
        o.to_frame(Galactic)

    # simple / with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = CartesianOrbit(pos=x, vel=v)
    coo,vel = o.to_frame(Galactic)
    assert coo.name == 'galactic'

def test_w():
    # simple / unitless
    x = np.random.random(size=(3,10))
    v = np.random.random(size=(3,10))
    o = CartesianOrbit(pos=x, vel=v)
    w = o.w()
    assert w.shape == (6,10)

    # simple / with units
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = CartesianOrbit(pos=x, vel=v)
    with pytest.raises(ValueError):
        o.w()
    w = o.w(units=galactic)
    assert np.allclose(x.value, w[:3])
    assert np.allclose(v.value, (w[3:]*u.kpc/u.Myr).to(u.km/u.s).value)

    # simple / with units and potential
    p = HernquistPotential(units=galactic, m=1E11, c=0.25)
    x = np.random.random(size=(3,10))*u.kpc
    v = np.random.normal(0.,100.,size=(3,10))*u.km/u.s
    o = CartesianOrbit(pos=x, vel=v, potential=p)
    w = o.w()
    assert np.allclose(x.value, w[:3])
    assert np.allclose(v.value, (w[3:]*u.kpc/u.Myr).to(u.km/u.s).value)

    w = o.w(units=solarsystem)
    assert np.allclose(x.value, (w[:3]*u.au).to(u.kpc).value)
    assert np.allclose(v.value, (w[3:]*u.au/u.yr).to(u.km/u.s).value)

