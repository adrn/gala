"""Astropy coordinate class for the Sagittarius coordinate system"""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix

__all__ = ["SagittariusLaw10", "SagittariusVasiliev21"]


class SagittariusLaw10(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Sagittarius dwarf galaxy, as described in
    Law & Majewski (2010): http://adsabs.harvard.edu/abs/2010ApJ...714..229L

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other
        keywords).

    Lambda : `~astropy.coordinates.Angle`, optional, must be keyword
        The longitude-like angle corresponding to Sagittarius' orbit.
    Beta : `~astropy.coordinates.Angle`, optional, must be keyword
        The latitude-like angle corresponding to Sagittarius' orbit.
    distance : `~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    pm_Lambda_cosBeta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion in ``Beta`` for this object (``pm_Lambda_cosBeta`` must
        also be given).
    radial_velocity : `~astropy.units.Quantity`, optional, must be keyword
        The radial velocity of this object.

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping("lon", "Lambda"),
            coord.RepresentationMapping("lat", "Beta"),
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


# Define the Euler angles (from Law & Majewski 2010)
phi = (180 + 3.75) * u.degree
theta = (90 - 13.46) * u.degree
psi = (180 + 14.111534) * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(phi, "z")
C = rotation_matrix(theta, "x")
B = rotation_matrix(psi, "z")
A = np.diag([1.0, 1.0, -1.0])
R = A @ B @ C @ D


# Galactic to Sgr coordinates
@frame_transform_graph.transform(
    coord.StaticMatrixTransform, coord.Galactic, SagittariusLaw10
)
def galactic_to_sgr():
    """Compute the transformation from Galactic spherical to
    heliocentric Sagittarius coordinates.
    """
    return R


# Sgr to Galactic coordinates
@frame_transform_graph.transform(
    coord.StaticMatrixTransform, SagittariusLaw10, coord.Galactic
)
def sgr_to_galactic():
    """Compute the transformation from heliocentric Sagittarius coordinates to
    spherical Galactic.
    """
    return galactic_to_sgr().T


# -------------------------------------------------------------------------------------


class SagittariusVasiliev21(coord.BaseCoordinateFrame):
    """
    A Heliocentric, right-handed spherical coordinate system defined by the orbit of the
    Sagittarius dwarf galaxy, as described in Vasiliev et al. (2021):
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.2279V/abstract

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other
        keywords).

    Lambda : `~astropy.coordinates.Angle`, optional, must be keyword
        The longitude-like angle corresponding to Sagittarius' orbit.
    Beta : `~astropy.coordinates.Angle`, optional, must be keyword
        The latitude-like angle corresponding to Sagittarius' orbit.
    distance : `~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    pm_Lambda_cosBeta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion along the stream in ``Lambda`` (including the
        ``cos(Beta)`` factor) for this object (``pm_Beta`` must also be given).
    pm_Beta : `~astropy.units.Quantity`, optional, must be keyword
        The proper motion in ``Beta`` for this object (``pm_Lambda_cosBeta`` must
        also be given).
    radial_velocity : `~astropy.units.Quantity`, optional, must be keyword
        The radial velocity of this object.

    """

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping("lon", "Lambda"),
            coord.RepresentationMapping("lat", "Beta"),
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


# Galactic to Sgr coordinates
@frame_transform_graph.transform(
    coord.StaticMatrixTransform, coord.Galactic, SagittariusVasiliev21
)
def galactic_to_sgr_v21():
    """Compute the transformation from Galactic to Sagittarius coordinates"""
    if not hasattr(SagittariusVasiliev21, "_R"):
        R = np.diag([1.0, -1.0, -1.0]) @ B @ C @ D
        SagittariusVasiliev21._R = R

    return SagittariusVasiliev21._R


# Sgr to Galactic coordinates
@frame_transform_graph.transform(
    coord.StaticMatrixTransform, SagittariusVasiliev21, coord.Galactic
)
def sgr_to_galactic_v21():
    """Compute the transformation from Sagittarius coordinates to Galactic coordinates"""
    return galactic_to_sgr_v21().T
