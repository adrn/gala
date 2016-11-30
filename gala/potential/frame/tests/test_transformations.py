# coding: utf-8

# Third-party
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import numpy as np
import pytest

# Project
from ..builtin import StaticFrame, ConstantRotatingFrame
from ..builtin.transformations import (static_to_constant_rotating,
                                       constant_rotating_to_static,
                                       rodrigues_axis_angle_rotate)
from ....dynamics import CartesianOrbit, CartesianPhaseSpacePosition
from ....units import galactic, dimensionless

def test_axis_angle_rotate():

    for x in [np.random.random(size=(3,32)), np.random.random(size=(3,32,8))]:
        vec = np.random.random(size=(3,32))
        theta = np.random.random(size=(32,))
        out = rodrigues_axis_angle_rotate(x, vec, theta)
        assert out.shape == x.shape

        vec = np.random.random(size=(3,))
        theta = np.random.random(size=(32,))
        out = rodrigues_axis_angle_rotate(x, vec, theta)
        assert out.shape == x.shape

        vec = np.random.random(size=(3,))
        theta = np.random.random(size=(1,))
        out = rodrigues_axis_angle_rotate(x, vec, theta)
        assert out.shape == x.shape

def _helper(fi, fr, w, t=None):

    pos_r,vel_r = static_to_constant_rotating(fi, fr, w, t=t)
    w2 = CartesianOrbit(pos=pos_r, vel=vel_r, t=t)
    pos_i,vel_i = constant_rotating_to_static(fr, fi, w2, t=t)

    assert quantity_allclose(pos_i, w.pos)
    assert quantity_allclose(vel_i, w.vel)

    pos_i,vel_i = constant_rotating_to_static(fr, fi, w, t=t)
    w2 = CartesianOrbit(pos=pos_i, vel=vel_i, t=t)
    pos_r,vel_r = static_to_constant_rotating(fi, fr, w2, t=t)

    assert quantity_allclose(pos_r, w.pos)
    assert quantity_allclose(vel_r, w.vel)

def test_frame_transforms():
    frame_i = StaticFrame(units=galactic)
    frame_r = ConstantRotatingFrame(Omega=[0.112, 1.235, 0.8656]*u.rad/u.Myr,
                                    units=galactic)

    w = CartesianOrbit(pos=np.random.random(size=(3,32))*u.kpc,
                       vel=np.random.random(size=(3,32))*u.kpc/u.Myr,
                       t=np.linspace(0,1,32)*u.Myr)
    _helper(frame_i, frame_r, w, t=w.t)

    w = CartesianOrbit(pos=np.random.random(size=(3,32,8))*u.kpc,
                       vel=np.random.random(size=(3,32,8))*u.kpc/u.Myr,
                       t=np.linspace(0,1,32)*u.Myr)
    _helper(frame_i, frame_r, w, t=w.t)

    w = CartesianPhaseSpacePosition(pos=np.random.random(size=3)*u.kpc,
                                    vel=np.random.random(size=3)*u.kpc/u.Myr)
    with pytest.raises(ValueError):
        _helper(frame_i, frame_r, w)
    _helper(frame_i, frame_r, w, t=0.*u.Myr)
    _helper(frame_i, frame_r, w, t=0.)

