# Third-party
from astropy.utils.misc import isiterable
import numpy as np

# Gala
from gala.dynamics import Orbit
from gala.units import DimensionlessUnitSystem

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
    theta = np.array(theta).T[..., None]

    out = np.cos(theta)*x + np.sin(theta) * np.cross(vec, x) + \
        (1 - np.cos(theta)) * (vec * x).sum(axis=-1)[..., None] * vec

    return out.T


def z_angle_rotate(xy, theta):
    """
    Rotated the input vector or set of vectors `xy` by the angle `theta`.

    Parameters
    ----------
    xy : array_like
        The vector or array of vectors to transform. Must have shape


    """
    xy = np.array(xy).T
    theta = np.array(theta).T

    out = np.zeros_like(xy)
    out[..., 0] = np.cos(theta)*xy[..., 0] - np.sin(theta)*xy[..., 1]
    out[..., 1] = np.sin(theta)*xy[..., 0] + np.cos(theta)*xy[..., 1]

    return out.T


def _constantrotating_static_helper(frame_r, frame_i, w, t=None, sign=1.):
    # TODO: use representation arithmetic instead
    Omega = -frame_r.parameters['Omega'].decompose(frame_i.units).value

    if not isinstance(w, Orbit) and t is None:
        raise ValueError("Time array must be provided if not passing an "
                         "Orbit subclass.")

    if t is None:
        t = w.t

    elif not hasattr(t, 'unit'):
        t = t * frame_i.units['time']

    if t is None:
        raise ValueError('Time must be supplied either through the input '
                         'Orbit class instance or through the t argument.')
    t = t.decompose(frame_i.units).value

    # HACK: this is a little bit crazy...this makes it so that !=3D
    #   representations will work here
    if hasattr(w.pos, 'xyz'):
        pos = w.pos
        vel = w.vel
    else:
        cart = w.cartesian
        pos = cart.pos
        vel = cart.vel

    pos = pos.xyz.decompose(frame_i.units).value
    vel = vel.d_xyz.decompose(frame_i.units).value

    # get rotation angle, axis vs. time
    if isiterable(Omega):  # 3D
        vec = Omega / np.linalg.norm(Omega)
        theta = np.linalg.norm(Omega) * t

        x_i2r = rodrigues_axis_angle_rotate(pos, vec, sign*theta)
        v_i2r = rodrigues_axis_angle_rotate(vel, vec, sign*theta)

    else:  # 2D
        vec = Omega * np.array([0, 0, 1.])
        theta = sign * Omega * t

        x_i2r = z_angle_rotate(pos, theta)
        v_i2r = z_angle_rotate(vel, theta)

    return x_i2r * frame_i.units['length'], v_i2r * frame_i.units['length']/frame_i.units['time']


def static_to_constantrotating(frame_i, frame_r, w, t=None):
    """
    Transform from an inertial static frame to a rotating frame.

    Parameters
    ----------
    frame_i : `~gala.potential.StaticFrame`
    frame_r : `~gala.potential.ConstantRotatingFrame`
    w : `~gala.dynamics.PhaseSpacePosition`, `~gala.dynamics.Orbit`
    t : quantity_like (optional)
        Required if input coordinates are just a phase-space position.

    Returns
    -------
    pos : `~astropy.units.Quantity`
        Position in rotating frame.
    vel : `~astropy.units.Quantity`
        Velocity in rotating frame.
    """
    return _constantrotating_static_helper(frame_r=frame_r, frame_i=frame_i,
                                           w=w, t=t, sign=1.)


def constantrotating_to_static(frame_r, frame_i, w, t=None):
    """
    Transform from a constantly rotating frame to a static, inertial frame.

    Parameters
    ----------
    frame_i : `~gala.potential.StaticFrame`
    frame_r : `~gala.potential.ConstantRotatingFrame`
    w : `~gala.dynamics.PhaseSpacePosition`, `~gala.dynamics.Orbit`
    t : quantity_like (optional)
        Required if input coordinates are just a phase-space position.

    Returns
    -------
    pos : `~astropy.units.Quantity`
        Position in static, inertial frame.
    vel : `~astropy.units.Quantity`
        Velocity in static, inertial frame.
    """
    return _constantrotating_static_helper(frame_r=frame_r, frame_i=frame_i,
                                           w=w, t=t, sign=-1.)


def static_to_static(frame_r, frame_i, w, t=None):
    """
    No-op transform

    Parameters
    ----------
    frame_i : `~gala.potential.StaticFrame`
    frame_r : `~gala.potential.ConstantRotatingFrame`
    w : `~gala.dynamics.PhaseSpacePosition`, `~gala.dynamics.Orbit`
    t : quantity_like (optional)
        Required if input coordinates are just a phase-space position.

    Returns
    -------
    pos : `~astropy.units.Quantity`
        Position in static, inertial frame.
    vel : `~astropy.units.Quantity`
        Velocity in static, inertial frame.
    """
    tmp = [isinstance(frame_r.units, DimensionlessUnitSystem),
           isinstance(frame_i.units, DimensionlessUnitSystem)]
    if not all(tmp) and any(tmp):
        raise ValueError(
            "StaticFrame to StaticFrame transformations are only allowed if "
            "both unit systems are physical, or both are dimensionless.")
    return w.pos.xyz, w.vel.d_xyz
