""" Astropy coordinate class for the Sagittarius coordinate system """

# Third-party
import numpy as np

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph

__all__ = ["GD1Koposov10"]


class GD1Koposov10(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit of the GD1 stream,
    as described in Koposov et al. 2010 (see: `<http://arxiv.org/abs/0907.1085>`_).

    For more information about this class, see the Astropy documentation on coordinate
    frames in :mod:`~astropy.coordinates`.

    Parameters
    ----------
    representation : :class:`~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    phi1 : angle_like, optional, must be keyword
        The longitude-like angle corresponding to GD-1's orbit.
    phi2 : angle_like, optional, must be keyword
        The latitude-like angle corresponding to GD-1's orbit.
    distance : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.
    pm_phi1_cosphi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the longitude-like direction corresponding to
        the GD-1 stream's orbit.
    pm_phi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the latitude-like direction perpendicular to the
        GD-1 stream's orbit.
    radial_velocity : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping("lon", "phi1"),
            coord.RepresentationMapping("lat", "phi2"),
            coord.RepresentationMapping("distance", "distance"),
        ],
    }

    _default_wrap_angle = 180 * u.deg

    def __init__(self, *args, **kwargs):
        wrap = kwargs.pop("wrap_longitude", True)
        super().__init__(*args, **kwargs)
        if wrap and isinstance(
            self._data,
            (coord.UnitSphericalRepresentation, coord.SphericalRepresentation),
        ):
            self._data.lon.wrap_angle = self._default_wrap_angle

    # TODO: remove this. This is a hack required as of astropy v3.1 in order
    # to have the longitude components wrap at the desired angle
    def represent_as(self, base, s="base", in_frame_units=False):
        r = super().represent_as(base, s=s, in_frame_units=in_frame_units)
        if hasattr(r, "lon"):
            r.lon.wrap_angle = self._default_wrap_angle
        return r

    represent_as.__doc__ = coord.BaseCoordinateFrame.represent_as.__doc__


# Rotation matrix as defined in the Appendix of Koposov et al. (2010)
R = np.array(
    [
        [-0.4776303088, -0.1738432154, 0.8611897727],
        [0.510844589, -0.8524449229, 0.111245042],
        [0.7147776536, 0.4930681392, 0.4959603976],
    ]
)


@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.ICRS, GD1Koposov10)
def icrs_to_gd1():
    """
    Compute the transformation from Galactic spherical to heliocentric GD1 coordinates.
    """
    return R


@frame_transform_graph.transform(coord.StaticMatrixTransform, GD1Koposov10, coord.ICRS)
def gd1_to_icrs():
    """
    Compute the transformation from heliocentric GD1 coordinates to spherical Galactic.
    """
    return icrs_to_gd1().T
