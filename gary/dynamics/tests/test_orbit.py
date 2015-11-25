# coding: utf-8

""" Test ...  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import pytest

# Project
from ..orbit import *

def test_initialize():

    with pytest.raises(ValueError):
        x = np.random.random(size=(10,3))
        v = np.random.random(size=(8,3))
        CartesianOrbit(pos=x, vel=v)

    with pytest.raises(ValueError):
        x = np.random.random(size=(10,3))
        v = np.random.random(size=(10,3))
        t = np.arange(8)
        CartesianOrbit(pos=x, vel=v, t=t)

    x = np.random.random(size=(10,3))
    v = np.random.random(size=(10,3))
    o = CartesianOrbit(pos=x, vel=v)
    assert o.ndim == 3

    x = np.random.random(size=(10,2))
    v = np.random.random(size=(10,2))
    o = CartesianOrbit(pos=x, vel=v)
    assert o.ndim == 2

def test_slice():

    # simple
    x = np.random.random(size=(10,3))
    v = np.random.random(size=(10,3))
    o = CartesianOrbit(pos=x, vel=v)
    new_o = o[:5]
    assert new_o.pos.shape == (5,3)

    # 1d slice on 3d
    x = np.random.random(size=(100,8,3))
    v = np.random.random(size=(100,8,3))
    t = np.arange(x.shape[0])
    o = CartesianOrbit(pos=x, vel=v, t=t)
    new_o = o[:5]
    assert new_o.pos.shape == (5,8,3)
    assert new_o.t.shape == (5,)

    # 3d slice on 3d
    o = CartesianOrbit(pos=x, vel=v, t=t)
    new_o = o[:5,:4]
    assert new_o.pos.shape == (5,4,3)
    assert new_o.t.shape == (5,)
