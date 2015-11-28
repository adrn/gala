# coding: utf-8

""" Test ...  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.coordinates import SphericalRepresentation
import astropy.units as u
import numpy as np
import pytest

# Project
from ..orbit import *

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
    print(sph_pos)
    print(sph_vel)
