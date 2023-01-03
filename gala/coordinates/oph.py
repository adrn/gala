""" Astropy coordinate class for the Ophiuchus coordinate system """

# Third-party
import numpy as np

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph


__all__ = ["OphiuchusPriceWhelan16"]


class OphiuchusPriceWhelan16(coord.BaseCoordinateFrame):
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
        The longitude-like angle corresponding to Ophiuchus's orbit.
    phi2 : angle_like, optional, must be keyword
        The latitude-like angle corresponding to Ophiuchus's orbit.
    distance : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    pm_phi1_cosphi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the longitude-like direction corresponding to
        the Ophiuchus stream's orbit.
    pm_phi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the latitude-like direction perpendicular to the
        Ophiuchus stream's orbit.
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
        ]
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


# Rotation matrix
R = np.array(
    [
        [0.84922096554, 0.07001279040, 0.52337554476],
        [-0.27043653641, -0.79364259852, 0.54497294023],
        [0.45352820359, -0.60434231606, -0.65504391727],
    ]
)


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, coord.Galactic, OphiuchusPriceWhelan16
)
def gal_to_oph():
    """Compute the transformation from Galactic spherical to
    heliocentric Ophiuchus coordinates.
    """
    return R


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, OphiuchusPriceWhelan16, coord.Galactic
)
def oph_to_gal():
    """Compute the transformation from heliocentric Ophiuchus coordinates to
    spherical Galactic.
    """
    return gal_to_oph().T
