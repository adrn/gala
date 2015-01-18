# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u

# Project
# ...

__all__ = ['cartesian_to_spherical', 'cartesian_to_physicsspherical',
           'cartesian_to_cylindrical']

def _pos_to_repr(pos):

    if hasattr(pos, 'xyz'):  # Representation-like
        pos_repr = coord.CartesianRepresentation(pos.xyz)

    elif hasattr(pos, 'cartesian'):  # Frame-like
        pos_repr = pos.cartesian

    elif hasattr(pos, 'unit'):  # Quantity-like
        pos_repr = coord.CartesianRepresentation(pos)

    else:
        raise TypeError("Unsupported position type '{0}'. Position must be "
                        "an Astropy Representation or Frame, or a "
                        "Quantity instance".format(type(pos)))

    return pos_repr

def cartesian_to_spherical(pos, vel):
    r"""
    Convert a velocity in Cartesian coordinates to velocity components
    in spherical coordinates. This follows the naming convention used in
    :mod:`astropy.coordinates`: spherical coordinates consist of a distance,
    :math:`d`, a longitude, :math:`\phi`, in the range [0, 360] deg, and
    a latitude, :math:`\delta`, in the range [-90, 90] deg.

    The components of the output velocity all have units of velocity, i.e.,
    this is not used for transforming from a cartesian velocity to angular
    velocities, but rather the velocity vector components in Eq. 2 below.

    .. math::

        \boldsymbol{v} &= v_x\boldsymbol{\hat{x}} + v_y\boldsymbol{\hat{y}} + v_z\boldsymbol{\hat{z}}
        &= v_r\boldsymbol{\hat{d}} + v_\phi\boldsymbol{\hat{\phi}} + v_\delta\boldsymbol{\hat{\delta}}
        &= \dot{d}\boldsymbol{\hat{d}} + d\cos\delta \dot{\phi}\boldsymbol{\hat{\phi}} + d\dot{\delta}\boldsymbol{\hat{\delta}}

    Parameters
    ----------
    pos : :class:`astropy.units.Quantity`, :class:`astropy.coordinates.BaseCoordinateFrame`, :class:`astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`astropy.units.Quantity` with :class:`astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`astropy.units.Quantity` with :class:`astropy.units.dimensionless_unscaled`
        units if you are working in natural units. axis=0 is assumed to be the
        dimensionality axis, e.g., ``vx,vy,vz = vel`` should work.

    Returns
    -------
    vsph : :class:`astropy.units.Quantity`
        Array of spherical velocity components. Will have the same shape as the
        input velocity.

    Examples
    --------

    """

    # position in Cartesian and spherical
    car_pos = _pos_to_repr(pos)
    sph_pos = car_pos.represent_as(coord.SphericalRepresentation)

    if not hasattr(vel, 'unit'):
        raise TypeError("Unsupported velocity type '{}'. Velocity must be "
                        "an Astropy Quantity instance.".format(type(vel)))

    # get out spherical components
    d = sph_pos.distance
    dxy = np.sqrt(car_pos.x**2 + car_pos.y**2)

    vr = np.sum(car_pos.xyz * vel, axis=0) / d

    mu_lon = (car_pos.xyz[0]*vel[1] - vel[0]*car_pos.xyz[1]) / dxy**2
    vlon = mu_lon * d

    mu_lat = (car_pos.xyz[2]*(car_pos.xyz[0]*vel[0] + car_pos.xyz[1]*vel[1]) - dxy**2*vel[2]) / d**2 / dxy
    vlat = -mu_lat * d * np.cos(sph_pos.lat)

    vsph = np.zeros_like(vel)
    vsph[0] = vr
    vsph[1] = vlon
    vsph[2] = vlat
    return vsph

def cartesian_to_physicsspherical(pos, vel):
    r"""
    Convert a velocity in Cartesian coordinates to velocity components
    in spherical coordinates. This follows the naming convention used in
    :mod:`astropy.coordinates`: physics spherical coordinates consist of
    a radius, :math:`r`, a longitude, :math:`\phi`, in the range [0, 360]
    deg and a colatitude, :math:`\theta`, in the range [0, 180] deg,
    measured from the z-axis.

    The components of the output velocity all have units of velocity, i.e.,
    this is not used for transforming from a cartesian velocity to angular
    velocities, but rather the velocity vector components in Eq. 2 below.

    .. math::

        \boldsymbol{v} &= v_x\boldsymbol{\hat{x}} + v_y\boldsymbol{\hat{y}} + v_z\boldsymbol{\hat{z}}
        &= v_r\boldsymbol{\hat{r}} + v_\phi\boldsymbol{\hat{\phi}} + v_\theta\boldsymbol{\hat{\theta}}
        &= \dot{r}\boldsymbol{\hat{r}} + r\sin\theta \dot{\phi}\boldsymbol{\hat{\phi}} + r\dot{\theta}\boldsymbol{\hat{\theta}}

    Parameters
    ----------
    pos : :class:`astropy.units.Quantity`, :class:`astropy.coordinates.BaseCoordinateFrame`, :class:`astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`astropy.units.Quantity` with :class:`astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`astropy.units.Quantity` with :class:`astropy.units.dimensionless_unscaled`
        units if you are working in natural units. axis=0 is assumed to be the
        dimensionality axis, e.g., ``vx,vy,vz = vel`` should work.

    Returns
    -------
    vsph : :class:`astropy.units.Quantity`
        Array of spherical velocity components. Will have the same shape as the
        input velocity.

    Examples
    --------

    """

    # position in Cartesian and spherical
    car_pos = _pos_to_repr(pos)
    sph_pos = car_pos.represent_as(coord.PhysicsSphericalRepresentation)

    if not hasattr(vel, 'unit'):
        raise TypeError("Unsupported velocity type '{}'. Velocity must be "
                        "an Astropy Quantity instance.".format(type(vel)))

    # get out spherical components
    r = sph_pos.r
    dxy = np.sqrt(car_pos.x**2 + car_pos.y**2)

    vr = np.sum(car_pos.xyz * vel, axis=0) / r

    mu_lon = (car_pos.xyz[0]*vel[1] - vel[0]*car_pos.xyz[1]) / dxy**2
    vlon = mu_lon * r

    mu_lat = (car_pos.xyz[2]*(car_pos.xyz[0]*vel[0] + car_pos.xyz[1]*vel[1]) - dxy**2*vel[2]) / r**2 / dxy
    vlat = mu_lat * r * np.sin(sph_pos.theta)

    vsph = np.zeros_like(vel)
    vsph[0] = vr
    vsph[1] = vlon
    vsph[2] = vlat
    return vsph

def cartesian_to_cylindrical(pos, vel):
    r"""
    Convert a velocity in Cartesian coordinates to velocity components
    in cylindrical coordinates. This follows the naming convention used in
    :mod:`astropy.coordinates`: cylindrical coordinates consist of a radius,
    :math:`\rho`, an azimuthal angle, :math:`\phi`, in the range [0, 360] deg,
    and a z coordinate, :math:`z`.

    The components of the output velocity all have units of velocity, i.e.,
    this is not used for transforming from a cartesian velocity to angular
    velocities, but rather the velocity vector components in Eq. 2 below.

    .. math::

        \boldsymbol{v} &= v_x\boldsymbol{\hat{x}} + v_y\boldsymbol{\hat{y}} + v_z\boldsymbol{\hat{z}}
        &= v_\rho\boldsymbol{\hat{\rho}} + v_\phi\boldsymbol{\hat{\phi}} + v_z\boldsymbol{\hat{z}}
        &= \dot{\rho}\boldsymbol{\hat{\rho}} + \rho\dot{\phi}\boldsymbol{\hat{\phi}} + \dot{z}\boldsymbol{\hat{\theta}}

    Parameters
    ----------
    pos : :class:`astropy.units.Quantity`, :class:`astropy.coordinates.BaseCoordinateFrame`, :class:`astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`astropy.units.Quantity` with :class:`astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`astropy.units.Quantity` with :class:`astropy.units.dimensionless_unscaled`
        units if you are working in natural units. axis=0 is assumed to be the
        dimensionality axis, e.g., ``vx,vy,vz = vel`` should work.

    Returns
    -------
    vcyl : :class:`astropy.units.Quantity`
        Array of spherical velocity components. Will have the same shape as the
        input velocity.

    Examples
    --------

    """

    # position in Cartesian and spherical
    car_pos = _pos_to_repr(pos)
    cyl_pos = car_pos.represent_as(coord.CylindricalRepresentation)

    if not hasattr(vel, 'unit'):
        raise TypeError("Unsupported velocity type '{}'. Velocity must be "
                        "an Astropy Quantity instance.".format(type(vel)))

    # get out spherical components
    rho = cyl_pos.rho
    vrho = np.sum(car_pos.xyz[:2] * vel[:2], axis=0) / rho

    phidot = (car_pos.xyz[0]*vel[1] - vel[0]*car_pos.xyz[1]) / rho**2
    vphi = phidot * rho

    vcyl = np.zeros_like(vel)
    vcyl[0] = vrho
    vcyl[1] = vphi
    vcyl[2] = vel[2]
    return vcyl

def spherical_to_cartesian(pos, vel):
    pass
    v = [rv*np.cos(a)*np.cos(d) - D*np.sin(a)*mura_cosdec - D*np.cos(a)*np.sin(d)*mudec,
              rv*np.sin(a)*np.cos(d) + D*np.cos(a)*mura_cosdec - D*np.sin(a)*np.sin(d)*mudec,
              rv*np.sin(d) + D*np.cos(d)*mudec]

def physicsspherical_to_cartesian(pos, vel):
    pass

def cylindrical_to_cartesian(pos, vel):
    pass
