# coding: utf-8

""" Utilities for fitting orbits to stream data. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.coordinates as coord
from astropy.coordinates.angles import rotation_matrix
import astropy.units as u
uno = u.dimensionless_unscaled
import numpy as np
from scipy.optimize import minimize

# Project
from .coordinates import Quaternion

def _rotation_opt_func(qua_wxyz, xyz):
    """
    Given a quaternion vector ``(w,x,y,z)`` and the data in heliocentric,
    Cartesian coordinates, compute the cost function for the given rotation.
    """
    R = Quaternion(qua_wxyz).rotation_matrix
    sph = coord.CartesianRepresentation(R.dot(xyz)*uno)\
               .represent_as(coord.SphericalRepresentation)
    return np.sum(sph.lat.degree**2)

def compute_stream_rotation_matrix(coordinate, wxyz0=None, align_lon=False):
    """
    Compute the rotation matrix to go from the frame of the input
    coordinate to closely align the equator with the stream.

    Parameter
    ---------
    coordinate : :class:`astropy.coordinate.SkyCoord`, :class:`astropy.coordinate.BaseCoordinateFrame`
        The coordinates of the stream stars.
    wxyz0 : array_like (optional)
        Initial guess for the quaternion vector that represents the rotation.
    align_lon : bool (optional)
        Also rotate in longitude so the minimum longitude is 0.

    Returns
    -------
    R : :class:`~numpy.ndarray`
        A 3 by 3 rotation matrix (has shape ``(3,3)``) to convert heliocentric,
        Cartesian coordinates in the input coordinate frame to stream coordinates.
    """
    if wxyz0 is None:
        wxyz0 = Quaternion.random().wxyz

    res = minimize(_rotation_opt_func, x0=wxyz0, args=(coordinate.cartesian.xyz,))
    R = Quaternion(res.x).rotation_matrix

    if align_lon:
        new_xyz = R.dot(coordinate.cartesian.xyz.value)
        lon = np.arctan2(new_xyz[1], new_xyz[0])
        R2 = rotation_matrix(lon.min()*u.radian, 'z')
        R = R2*R

    return R

def rotate_sph_coordinate(rep, R):
    """

    Parameter
    ---------
    rep : :class:`astropy.coordinate.BaseCoordinateFrame`, :class:`astropy.coordinate.BaseRepresentation`
        The coordinates.
    R : array_like
        The rotation matrix.

    Returns
    -------
    sph : :class:`astropy.coordinates.SphericalRepresentation`

    """
    if hasattr(rep, 'xyz'):
        xyz = rep.xyz.value
        unit = rep.xyz.unit
    elif hasattr(rep, 'represent_as'):
        xyz = rep.represent_as(coord.CartesianRepresentation).xyz
        unit = xyz.unit
        xyz = xyz.value
    else:
        raise TypeError("Input rep must be either a BaseRepresentation or a "
                        "BaseCoordinateFrame subclass.")

    sph = coord.CartesianRepresentation(R.dot(xyz)*unit)\
               .represent_as(coord.SphericalRepresentation)
    return sph
