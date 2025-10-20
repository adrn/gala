"""Astropy coordinate class for the Orphan stream coordinate systems"""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix

__all__ = ["OrphanKoposov19", "OrphanNewberg10"]


class OrphanNewberg10(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Orphan stream, as described in
    Newberg et al. 2010 (see: `<http://arxiv.org/abs/1001.0576>`_).

    Note: to be consistent with other stream classes, we refer to the longitude
    and latitude as ``phi1`` and ``phi2`` instead of ``Lambda`` and ``Beta``.

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
        The radial velocity for this object along the line-of-sight.

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
            coord.UnitSphericalRepresentation | coord.SphericalRepresentation,
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


# Define the Euler angles
phi = 128.79 * u.degree
theta = 54.39 * u.degree
psi = 90.70 * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(phi, "z")
C = rotation_matrix(theta, "x")
B = rotation_matrix(psi, "z")
R = B @ C @ D


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, coord.Galactic, OrphanNewberg10
)
def galactic_to_orp():
    """Compute the transformation from Galactic spherical to
    heliocentric Orphan coordinates.
    """
    return R


# Oph to Galactic coordinates
@frame_transform_graph.transform(
    coord.StaticMatrixTransform, OrphanNewberg10, coord.Galactic
)
def orp_to_galactic():
    """Compute the transformation from heliocentric Orphan coordinates to
    spherical Galactic.
    """
    return galactic_to_orp().T


# ------------------------------------------------------------------------------


class OrphanKoposov19(coord.BaseCoordinateFrame):
    """A coordinate frame for the Orphan stream defined by Sergey Koposov.

    Parameters
    ----------
    phi1 : `~astropy.units.Quantity`
        Longitude component.
    phi2 : `~astropy.units.Quantity`
        Latitude component.
    distance : `~astropy.units.Quantity`
        Distance.

    pm_phi1_cosphi2 : `~astropy.units.Quantity`
        Proper motion in longitude.
    pm_phi2 : `~astropy.units.Quantity`
        Proper motion in latitude.
    radial_velocity : `~astropy.units.Quantity`
        Line-of-sight or radial velocity.
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
            coord.UnitSphericalRepresentation | coord.SphericalRepresentation,
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


@frame_transform_graph.transform(
    coord.StaticMatrixTransform, coord.ICRS, OrphanKoposov19
)
def icrs_to_orp19():
    """Compute the transformation from ICRS to
    heliocentric Orphan coordinates.
    """
    return np.array(
        [
            [-0.44761231, -0.08785756, -0.88990128],
            [-0.84246097, 0.37511331, 0.38671632],
            [0.29983786, 0.92280606, -0.2419219],
        ]
    )


# Oph to Galactic coordinates
@frame_transform_graph.transform(
    coord.StaticMatrixTransform, OrphanKoposov19, coord.ICRS
)
def orp19_to_icrs():
    """Compute the transformation from heliocentric Orphan coordinates to
    spherical ICRS.
    """
    return icrs_to_orp19().T
