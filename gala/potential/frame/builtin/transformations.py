# coding: utf-8

from __future__ import division, print_function

# Third-party
import numpy as np

# Gala
from ....dynamics import Orbit

__all__ = ['static_to_constantrotating', 'constantrotating_to_static']

def rodrigues_axis_angle_rotate(x, vec, theta):
    """
    Rotated the input vector or set of vectors `x` around the axis
    `vec` by the angle `theta`.

    Parameters
    ----------
    x : array_like
        The vector or array of vectors to transform. Must have shape


    """
    x = np.array(x).T
    vec = np.array(vec).T
    theta = np.array(theta).T[...,None]

    out = np.cos(theta)*x + np.sin(theta)*np.cross(vec, x) \
            + (1 - np.cos(theta)) * (vec * x).sum(axis=-1)[...,None] * vec

    return out.T

def static_to_constantrotating(frame_i, frame_r, w, t=None):
    """
    Transform from an inertial static frame to a rotating frame.

    Parameters
    ----------
    frame_i : `~gala.potential.StaticFrame`
    frame_r : `~gala.potential.ConstantRotatingFrame`
    w : `~gala.dynamics.CartesianPhaseSpacePosition`, `~gala.dynamics.CartesianOrbit`
    t : quantity_like (optional)
        Required if input coordinates are just a phase-space position.

    Returns
    -------
    pos : `~astropy.units.Quantity`
        Position in rotating frame.
    vel : `~astropy.units.Quantity`
        Velocity in rotating frame.
    """
    Omega = frame_r.Omega.decompose(frame_i.units).value

    if not isinstance(w, Orbit) and t is None:
        raise ValueError("Time array must be provided if not passing an "
                         "Orbit subclass.")

    if t is None:
        t = w.t.decompose(frame_i.units).value

    # get rotation angle, axis vs. time
    vec = -Omega / np.linalg.norm(Omega)
    theta = np.linalg.norm(Omega) * t

    pos = w.pos.decompose(frame_i.units).value
    vel = w.vel.decompose(frame_i.units).value

    x_i2r = rodrigues_axis_angle_rotate(pos, vec, theta)
    v_i2r = rodrigues_axis_angle_rotate(vel, vec, theta)

    return x_i2r * frame_i.units['length'], v_i2r * frame_i.units['length']/frame_i.units['time']

def constantrotating_to_static(frame_r, frame_i, w, t=None):
    """
    Transform from a constantly rotating frame to a static, inertial frame.

    Parameters
    ----------
    frame_i : `~gala.potential.StaticFrame`
    frame_r : `~gala.potential.ConstantRotatingFrame`
    w : `~gala.dynamics.CartesianPhaseSpacePosition`, `~gala.dynamics.CartesianOrbit`
    t : quantity_like (optional)
        Required if input coordinates are just a phase-space position.

    Returns
    -------
    pos : `~astropy.units.Quantity`
        Position in static, inertial frame.
    vel : `~astropy.units.Quantity`
        Velocity in static, inertial frame.
    """
    Omega = -frame_r.Omega.decompose(frame_i.units).value

    if not isinstance(w, Orbit) and t is None:
        raise ValueError("Time array must be provided if not passing an "
                         "Orbit subclass.")

    if t is None:
        t = w.t.decompose(frame_i.units).value

    # get rotation angle, axis vs. time
    vec = Omega / np.linalg.norm(Omega)
    theta = np.linalg.norm(Omega) * t

    pos = w.pos.decompose(frame_i.units).value
    vel = w.vel.decompose(frame_i.units).value

    x_r2i = rodrigues_axis_angle_rotate(pos, vec, -theta)
    v_r2i = rodrigues_axis_angle_rotate(vel, vec, -theta)

    return x_r2i * frame_i.units['length'], v_r2i * frame_i.units['length']/frame_i.units['time']
