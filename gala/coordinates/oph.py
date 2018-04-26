# coding: utf-8

""" Astropy coordinate class for the Ophiuchus coordinate system """

# Third-party
import numpy as np

import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose

__all__ = ["Ophiuchus"]

class Ophiuchus(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Ophiuchus stream, as described in
    Price-Whelan et al. 2016 (see: `<https://arxiv.org/abs/1601.06790>`_).

    For more information about this class, see the Astropy documentation
    on coordinate frames in :mod:`~astropy.coordinates`.

    Parameters
    ----------
    representation : :class:`~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)

    phi1 : angle_like, optional, must be keyword
        The longitude-like angle corresponding to Orphan's orbit.
    phi2 : angle_like, optional, must be keyword
        The latitude-like angle corresponding to Orphan's orbit.
    distance : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    pm_phi1_cosphi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the longitude-like direction corresponding to
        the Orphan stream's orbit.
    pm_phi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the latitude-like direction perpendicular to the
        Orphan stream's orbit.
    radial_velocity : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    """
    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'phi1'),
            coord.RepresentationMapping('lat', 'phi2'),
            coord.RepresentationMapping('distance', 'distance')],
        coord.SphericalCosLatDifferential: [
            coord.RepresentationMapping('d_lon_coslat', 'pm_phi1_cosphi2'),
            coord.RepresentationMapping('d_lat', 'pm_phi2'),
            coord.RepresentationMapping('d_distance', 'radial_velocity')],
        coord.SphericalDifferential: [
            coord.RepresentationMapping('d_lon', 'pm_phi1'),
            coord.RepresentationMapping('d_lat', 'pm_phi2'),
            coord.RepresentationMapping('d_distance', 'radial_velocity')]
    }

    frame_specific_representation_info[coord.UnitSphericalRepresentation] = \
        frame_specific_representation_info[coord.SphericalRepresentation]
    frame_specific_representation_info[coord.UnitSphericalCosLatDifferential] = \
        frame_specific_representation_info[coord.SphericalCosLatDifferential]
    frame_specific_representation_info[coord.UnitSphericalDifferential] = \
        frame_specific_representation_info[coord.SphericalDifferential]

# Rotation matrix as defined in the Appendix of Koposov et al. (2010)
R = np.array([[0.84922096554, 0.07001279040, 0.52337554476],
              [-0.27043653641 -0.79364259852 0.54497294023],
              [0.45352820359 -0.60434231606 -0.65504391727]])

@frame_transform_graph.transform(coord.StaticMatrixTransform,
                                 coord.ICRS, Ophiuchus)
def icrs_to_oph():
    """ Compute the transformation from Galactic spherical to
        heliocentric Ophiuchus coordinates.
    """
    return R

@frame_transform_graph.transform(coord.StaticMatrixTransform,
                                 Ophiuchus, coord.ICRS)
def oph_to_icrs():
    """ Compute the transformation from heliocentric Ophiuchus coordinates to
        spherical Galactic.
    """
    return matrix_transpose(icrs_to_gd1())
