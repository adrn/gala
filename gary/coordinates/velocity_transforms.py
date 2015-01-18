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

__all__ = ['cartesian_to_spherical']

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
    """
    Convert a velocity in Cartesian coordinates to velocity components
    in spherical coordinates. This follows the naming convention used in
    :mod:`astropy.coordinates`: spherical coordinates consist of a longitude
    in the range [0,360) deg and latitude in the range [-90, 90] deg.

    Parameters
    ----------
    pos : :class:`astropy.units.Quantity`, :class:`astropy.coordinates.BaseCoordinateFrame`, :class:`astropy.coordinates.BaseRepresentation`
        Input position or positions as one of the allowed types. You may pass in a
        :class:`astropy.units.Quantity` with :class:`astropy.units.dimensionless_unscaled`
        units if you are working in natural units.
    vel : :class:`astropy.units.Quantity`
        Input velocity or velocities as one of the allowed types. You may pass in a
        :class:`astropy.units.Quantity` with :class:`astropy.units.dimensionless_unscaled`
        units if you are working in natural units.

    Returns
    -------
    vsph : :class:`astropy.units.Quantity`

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

    mu_lon = -(car_pos.xyz[0]*vel[1] - vel[0]*car_pos.xyz[1]) / dxy**2
    vlon = mu_lon * d

    mu_lat = -(car_pos.xyz[2]*(car_pos.xyz[0]*vel[0] + car_pos.xyz[1]*vel[1]) - dxy**2*vel[2]) / d**2 / dxy
    vlat = mu_lat * d * np.cos(sph_pos.lat)

    vsph = np.zeros_like(vel)
    vsph[0] = vr
    vsph[1] = vlon
    vsph[2] = vlat
    return vsph

def spherical_to_cartesian(pos, vel):
    pass
    v = [rv*np.cos(a)*np.cos(d) - D*np.sin(a)*mura_cosdec - D*np.cos(a)*np.sin(d)*mudec,
              rv*np.sin(a)*np.cos(d) + D*np.cos(a)*mura_cosdec - D*np.sin(a)*np.sin(d)*mudec,
              rv*np.sin(d) + D*np.cos(d)*mudec]

def cartesian_to_physicsspherical(pos, vel):
    pass

def physicsspherical_to_cartesian(pos, vel):
    pass

def cartesian_to_cylindrical(pos, vel):
    pass

def cylindrical_to_cartesian(pos, vel):
    pass
