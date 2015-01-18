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

def cartesian_to_spherical(pos, vel):
    """
    Convert a velocity in Cartesian coordinates to velocity components
    in spherical coordinates. This follows the naming convention used in
    :mod:`astropy.coordinates`: spherical coordinates consist of a longitude
    in the range [0,360) deg and latitude in the range [-90, 90] deg.

    Parameters
    ----------
    pos : array_like, :class:`astropy.coordinates.BaseCoordinateFrame`
    vel :

    Returns
    -------

    Examples
    --------

    """

    # if this is an astropy coordinate frame, need to get out cartesian pos.
    try:
        xyz = pos.cartesian.xyz
    except AttributeError:
        xyz = pos

    # if it doesn't have a unit, make it dimensionless_unscaled
    try:
        xyz.unit
    except AttributeError:
        xyz = xyz * u.dimensionless_unscaled

    # convert position to spherical
    car_pos = coord.CartesianRepresentation(xyz)
    sph_pos = car_pos.represent_as(coord.SphericalRepresentation)

    # get out spherical components
    d = sph_pos.distance
    dxy = np.sqrt(car_pos.x**2 + car_pos.y**2)

    vr = np.sum(car_pos.xyz * vel, axis=0) / d
    mu_lon = -(car_pos.xyz[0]*vel[1] - vel[0]*car_pos.xyz[1]) / dxy**2
    mu_lat = -(car_pos.xyz[2]*(car_pos.xyz[0]*vel[0] + car_pos.xyz[1]*vel[1]) - dxy**2*vel[2]) / d**2 / dxy

    return (vr, mu_lon, mu_lat)

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
