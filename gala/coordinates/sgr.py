# coding: utf-8

""" Astropy coordinate class for the Sagittarius coordinate system """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from numpy import cos, sin

from astropy.coordinates import frame_transform_graph
from astropy.coordinates.angles import rotation_matrix
import astropy.coordinates as coord
import astropy.units as u

__all__ = ["Sagittarius"]

class Sagittarius(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Sagittarius dwarf galaxy, as described in
    Majewski et al. 2003 (see: `<http://adsabs.harvard.edu/abs/2003ApJ...599.1082M>`_)
    and further explained at
    `this website <http://www.astro.virginia.edu/~srm4n/Sgr/>`_.

    For more information about this class, see the Astropy documentation
    on `Coordinate Frames <http://docs.astropy.org/en/latest/coordinates/frames.html>`_.

    Parameters
    ----------
    representation : :class:`~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    Lambda : angle_like, optional, must be keyword
        The longitude-like angle corresponding to Sagittarius' orbit.
    Beta : angle_like, optional, must be keyword
        The latitude-like angle corresponding to Sagittarius' orbit.
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

# Define the Euler angles (from Law & Majewski 2010)
phi = np.radians(180+3.75)
theta = np.radians(90-13.46)
psi = np.radians(180+14.111534)

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(phi, "z", unit=u.radian)
C = rotation_matrix(theta, "x", unit=u.radian)
B = rotation_matrix(psi, "z", unit=u.radian)
A = np.diag([1.,1.,-1.])
R = np.array(A.dot(B).dot(C).dot(D))

# Galactic to Sgr coordinates
@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Sagittarius)
def galactic_to_sgr():
    """ Compute the transformation from Galactic spherical to
        heliocentric Orphan coordinates.
    """
    return R

# Sgr to Galactic coordinates
@frame_transform_graph.transform(coord.StaticMatrixTransform, Sagittarius, coord.Galactic)
def sgr_to_galactic():
    """ Compute the transformation from heliocentric Orphan coordinates to
        spherical Galactic.
    """
    return galactic_to_sgr().T
