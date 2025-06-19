# Third-party
import numpy as np

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph

import numpy as np
from astropy.coordinates import ICRS, StaticMatrixTransform, frame_transform_graph
from astropy.utils.decorators import format_doc
from ._builtin_frames.base import BaseStreamFrame, stream_doc


__all__ = ["JhelumBonaca19", "JhelumShipp19"]


class JhelumBonaca19(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit of the Jhelum
    stream, as described in Bonaca et al. 2019.

    For more information about this class, see the Astropy documentation on coordinate
    frames in :mod:`~astropy.coordinates`.

    Parameters
    ----------
    representation : :class:`~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    phi1 : angle_like, optional, must be keyword
        The longitude-like angle aligned with the stream.
    phi2 : angle_like, optional, must be keyword
        The latitude-like angle aligned perpendicular to the stream.
    distance : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    pm_phi1_cosphi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the longitude-like direction corresponding to
        the Jhelum stream's orbit.
    pm_phi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the latitude-like direction perpendicular to the
        Jhelum stream's orbit.
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


# Rotation matrix as defined in Bonaca+2019
R = np.array(
    [
        [0.6173151074, -0.0093826715, -0.7866600433],
        [-0.0151801852, -0.9998847743, 0.0000135163],
        [-0.7865695266, 0.0119333013, -0.6173864075],
    ]
)


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, coord.ICRS, JhelumBonaca19
)
def icrs_to_jhelum():
    """
    Compute the transformation from Galactic spherical to heliocentric Jhelum
    coordinates.
    """
    return R


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, JhelumBonaca19, coord.ICRS
)
def jhelum_to_icrs():
    """
    Compute the transformation from heliocentric Jhelum coordinates to spherical ICRS.
    """
    return icrs_to_jhelum().T


# =============================================================================



@format_doc(
    stream_doc,
    name="Jhelum",
    paper="Shipp et al. 2019",
    url="https://ui.adsabs.harvard.edu/abs/2019ApJ...885....3S",
)
class JhelumShipp19(BaseStreamFrame):
    """A coordinate system defined by the orbit of the Jhelum stream."""


R = np.array(
    [
        [0.60334991, -0.20211605, -0.77143890],
        [-0.13408072, -0.97928924, 0.15170675],
        [0.78612419, -0.01190283, 0.61795395],
    ]
)


@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.ICRS, JhelumShipp19)
def icrs_to_indus():
    return R


@frame_transform_graph.transform(coord.StaticMatrixTransform, JhelumShipp19, coord.ICRS)
def indus_to_icrs():
    return R.T
