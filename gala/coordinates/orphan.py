# coding: utf-8

""" Astropy coordinate class for the Sagittarius coordinate system """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
import astropy.coordinates as coord
import astropy.units as u

__all__ = ["Orphan"]

class Orphan(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Orphan stream, as described in
    Newberg et al. 2010 (see: `<http://arxiv.org/abs/1001.0576>`_).

    For more information about this class, see the Astropy documentation
    on coordinate frames in :mod:`~astropy.coordinates`.

    Parameters
    ----------
    representation : :class:`~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    Lambda : angle_like, optional, must be keyword
        The longitude-like angle corresponding to Orphan's orbit.
    Beta : angle_like, optional, must be keyword
        The latitude-like angle corresponding to Orphan's orbit.
    distance : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    """
    default_representation = coord.SphericalRepresentation

    frame_specific_representation_info = {
        'spherical': [coord.RepresentationMapping('lon', 'Lambda'),
                      coord.RepresentationMapping('lat', 'Beta'),
                      coord.RepresentationMapping('distance', 'distance')],
        'unitspherical': [coord.RepresentationMapping('lon', 'Lambda'),
                          coord.RepresentationMapping('lat', 'Beta')]
    }

# Define the Euler angles
phi = 128.79 * u.degree
theta = 54.39 * u.degree
psi = 90.70 * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(phi, "z")
C = rotation_matrix(theta, "x")
B = rotation_matrix(psi, "z")
R = matrix_product(B, C, D)

@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Orphan)
def galactic_to_orp():
    """ Compute the transformation from Galactic spherical to
        heliocentric Orphan coordinates.
    """
    return R

# Oph to Galactic coordinates
@frame_transform_graph.transform(coord.StaticMatrixTransform, Orphan, coord.Galactic)
def oph_to_galactic():
    """ Compute the transformation from heliocentric Orphan coordinates to
        spherical Galactic.
    """
    return matrix_transpose(galactic_to_orp())
