import astropy.units as u
import numpy as np
import pytest

from gala.dynamics import Orbit, PhaseSpacePosition
from gala.potential import ConstantRotatingFrame, StaticFrame
from gala.potential.frame.builtin.transformations import (
    constantrotating_to_static,
    rodrigues_axis_angle_rotate,
    static_to_constantrotating,
)
from gala.units import galactic


def test_axis_angle_rotate():
    rng = np.random.default_rng(42)
    for x in [rng.random(size=(3, 32)), rng.random(size=(3, 32, 8))]:
        vec = rng.random(size=(3, 32))
        theta = rng.random(size=(32,))
        out = rodrigues_axis_angle_rotate(x, vec, theta)
        assert out.shape == x.shape

        vec = rng.random(size=(3,))
        theta = rng.random(size=(32,))
        out = rodrigues_axis_angle_rotate(x, vec, theta)
        assert out.shape == x.shape

        vec = rng.random(size=(3,))
        theta = rng.random(size=(1,))
        out = rodrigues_axis_angle_rotate(x, vec, theta)
        assert out.shape == x.shape


def _helper(fi, fr, w, t=None):
    pos_r, vel_r = static_to_constantrotating(fi, fr, w, t=t)
    if isinstance(w, Orbit):
        w2 = Orbit(pos=pos_r, vel=vel_r, t=t)
    else:
        w2 = PhaseSpacePosition(pos=pos_r, vel=vel_r)
    pos_i, vel_i = constantrotating_to_static(fr, fi, w2, t=t)

    assert u.allclose(pos_i, w.xyz)
    assert u.allclose(vel_i, w.v_xyz)

    pos_i, vel_i = constantrotating_to_static(fr, fi, w, t=t)
    if isinstance(w, Orbit):
        w2 = Orbit(pos=pos_i, vel=vel_i, t=t)
    else:
        w2 = PhaseSpacePosition(pos=pos_i, vel=vel_i)
    pos_r, vel_r = static_to_constantrotating(fi, fr, w2, t=t)

    assert u.allclose(pos_r, w.xyz)
    assert u.allclose(vel_r, w.v_xyz)


def test_frame_transforms_3d():
    rng = np.random.default_rng(42)
    frame_i = StaticFrame(units=galactic)
    frame_r = ConstantRotatingFrame(
        Omega=[0.112, 1.235, 0.8656] * u.rad / u.Myr, units=galactic
    )

    w = Orbit(
        pos=rng.random(size=(3, 32)) * u.kpc,
        vel=rng.random(size=(3, 32)) * u.kpc / u.Myr,
        t=np.linspace(0, 1, 32) * u.Myr,
    )
    _helper(frame_i, frame_r, w, t=w.t)

    w = Orbit(
        pos=rng.random(size=(3, 32, 8)) * u.kpc,
        vel=rng.random(size=(3, 32, 8)) * u.kpc / u.Myr,
        t=np.linspace(0, 1, 32) * u.Myr,
    )
    _helper(frame_i, frame_r, w, t=w.t)

    w = PhaseSpacePosition(
        pos=rng.random(size=3) * u.kpc,
        vel=rng.random(size=3) * u.kpc / u.Myr,
    )
    with pytest.raises(ValueError):
        _helper(frame_i, frame_r, w)
    _helper(frame_i, frame_r, w, t=0.0 * u.Myr)
    _helper(frame_i, frame_r, w, t=0.0)


def test_frame_transforms_2d():
    rng = np.random.default_rng(42)
    frame_i = StaticFrame(units=galactic)
    frame_r = ConstantRotatingFrame(Omega=0.529 * u.rad / u.Myr, units=galactic)

    w = Orbit(
        pos=rng.random(size=(2, 32)) * u.kpc,
        vel=rng.random(size=(2, 32)) * u.kpc / u.Myr,
        t=np.linspace(0, 1, 32) * u.Myr,
    )
    _helper(frame_i, frame_r, w, t=w.t)

    w = Orbit(
        pos=rng.random(size=(2, 32, 8)) * u.kpc,
        vel=rng.random(size=(2, 32, 8)) * u.kpc / u.Myr,
        t=np.linspace(0, 1, 32) * u.Myr,
    )
    _helper(frame_i, frame_r, w, t=w.t)

    w = PhaseSpacePosition(
        pos=rng.random(size=2) * u.kpc,
        vel=rng.random(size=2) * u.kpc / u.Myr,
    )
    with pytest.raises(ValueError):
        _helper(frame_i, frame_r, w)
    _helper(frame_i, frame_r, w, t=0.0 * u.Myr)
    _helper(frame_i, frame_r, w, t=0.0)
