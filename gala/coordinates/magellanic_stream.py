# coding: utf-8

""" Astropy coordinate class for the Magellanic Stream coordinate system """

from __future__ import division, print_function

from astropy.coordinates.matrix_utilities import (rotation_matrix,
                                                  matrix_product,
                                                  matrix_transpose)
from astropy.coordinates.baseframe import (frame_transform_graph,
                                           BaseCoordinateFrame,
                                           RepresentationMapping)
from astropy.coordinates.transformations import StaticMatrixTransform
from astropy.coordinates import representation as r
from astropy.coordinates import Galactic

import astropy.units as u

__all__ = ["MagellanicStream"]


class MagellanicStream(BaseCoordinateFrame):
    """
    A coordinate or frame aligned with the Magellanic Stream,
    as defined by Nidever et al. (2008,
    see: `<http://adsabs.harvard.edu/abs/2008ApJ...679..432N>`_).

    For more information about this class, see the Astropy documentation
    on coordinate frames in :mod:`~astropy.coordinates`.

    Example
    -------
    Converting the coordinates of the Large Magellanic Cloud:

        >>> from astropy import coordinates as coord
        >>> from astropy import units as u
        >>> from gala.coordinates import MagellanicStream

        >>> c = coord.Galactic(l=280.4652*u.deg, b=-32.8884*u.deg)
        >>> ms = c.transform_to(MagellanicStream)
        >>> print(ms)
        <MagellanicStream Coordinate: (L, B) in deg
            (359.86313884, 2.42583948)>
    """

    frame_specific_representation_info = {
        r.SphericalRepresentation: [
            RepresentationMapping('lon', 'L'),
            RepresentationMapping('lat', 'B')
        ]
    }

    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential

    _ngp = Galactic(l=188.5*u.deg, b=-7.5*u.deg)
    _lon0 = Galactic(l=280.47*u.deg, b=-32.75*u.deg)


@frame_transform_graph.transform(StaticMatrixTransform,
                                 Galactic, MagellanicStream)
def gal_to_mag():
    mat1 = rotation_matrix(57.275785782128686*u.deg, 'z')
    mat2 = rotation_matrix(90*u.deg - MagellanicStream._ngp.b, 'y')
    mat3 = rotation_matrix(MagellanicStream._ngp.l, 'z')

    return matrix_product(mat1, mat2, mat3)


@frame_transform_graph.transform(StaticMatrixTransform,
                                 MagellanicStream, Galactic)
def mag_to_gal():
    return matrix_transpose(gal_to_mag())
