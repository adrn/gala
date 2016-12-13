# coding: utf-8

""" Astropy coordinate class for the Sagittarius coordinate system """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

from astropy.coordinates import frame_transform_graph
import astropy.coordinates as coord
import astropy.units as u

try:
    from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
    ASTROPY_1_3 = True
except ImportError:
    from .matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
    ASTROPY_1_3 = False

if not ASTROPY_1_3:
    import astropy
    import warnings
    warnings.warn("We recommend using Astropy v1.3 or later. You have: {}"
                  .format(astropy.__version__), DeprecationWarning)

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
phi = (180+3.75) * u.degree
theta = (90-13.46) * u.degree
psi = (180+14.111534) * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(phi, "z")
C = rotation_matrix(theta, "x")
B = rotation_matrix(psi, "z")
A = np.diag([1.,1.,-1.])
R = matrix_product(A, B, C, D)

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
    return matrix_transpose(galactic_to_sgr())
