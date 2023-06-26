"""Base class for Stream Coordinate Frames."""

from __future__ import annotations

__all__: list[str] = []

from astropy.coordinates import BaseCoordinateFrame
from astropy.utils.decorators import format_doc
from astropy.coordinates import (
    BaseCoordinateFrame,
    RepresentationMapping,
    SphericalRepresentation,
    UnitSphericalCosLatDifferential,
    UnitSphericalRepresentation,
)


stream_doc = """A coordinate system defined by the orbit of the {name} stream.

For more information about the stream see {paper}
(`<{url}>`_).

For more information about this class, see the Astropy documentation on
coordinate frames in :mod:`~astropy.coordinates`.

Parameters
----------
representation : :class:`~astropy.coordinates.BaseRepresentation` or None
    A representation object or None to have no data (or use the other
    keywords)
phi1 : angle_like, optional, must be keyword
    The longitude-like angle corresponding to {name}'s orbit.
phi2 : angle_like, optional, must be keyword
    The latitude-like angle corresponding to {name}'s orbit.
distance : :class:`~astropy.units.Quantity`, optional, must be keyword
    The Distance for this object along the line-of-sight.
pm_phi1_cosphi2 : :class:`~astropy.units.Quantity`, optional, must be
keyword
    The proper motion in the longitude-like direction corresponding to the
    {name} stream's orbit.
pm_phi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
    The proper motion in the latitude-like direction perpendicular to the
    {name} stream's orbit.
radial_velocity : :class:`~astropy.units.Quantity`, optional, must be
keyword
    The Distance for this object along the line-of-sight.
"""


@format_doc(
    stream_doc,
    name="BaseStreamFrame",
    paper="Price-Whelan et al. 2020",
    url="http://gala.adrian.pw/en/latest/",
)
class BaseStreamFrame(BaseCoordinateFrame):
    default_representation = UnitSphericalRepresentation
    default_differential = UnitSphericalCosLatDifferential

    frame_specific_representation_info = {
        UnitSphericalRepresentation: [
            RepresentationMapping("lon", "phi1"),
            RepresentationMapping("lat", "phi2"),
        ],
        SphericalRepresentation: [
            RepresentationMapping("lon", "phi1"),
            RepresentationMapping("lat", "phi2"),
        ],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.has_data and hasattr(self.data, "lon"):
            self.data.lon.wrap_angle = 180.0 * u.deg
