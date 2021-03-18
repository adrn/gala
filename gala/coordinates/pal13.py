""" Astropy coordinate class for the Palomar 5 stream coordinate system """

# Third-party
import numpy as np

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose

__all__ = ["Pal13Shipp20"]


class Pal13Shipp20(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Pal 13 stream by Shipp et al. (2020).

    For more information about this class, see the Astropy documentation
    on coordinate frames in :mod:`~astropy.coordinates`.

    Parameters
    ----------
    representation : :class:`~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)

    phi1 : angle_like, optional, must be keyword
        The longitude-like angle corresponding to Pal 13's orbit.
    phi2 : angle_like, optional, must be keyword
        The latitude-like angle corresponding to Pal 13's orbit.
    distance : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    pm_phi1_cosphi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the longitude-like direction corresponding to
        the Pal 5 stream's orbit.
    pm_phi2 : :class:`~astropy.units.Quantity`, optional, must be keyword
        The proper motion in the latitude-like direction perpendicular to the
        Pal 5 stream's orbit.
    radial_velocity : :class:`~astropy.units.Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    """
    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'phi1'),
            coord.RepresentationMapping('lat', 'phi2'),
            coord.RepresentationMapping('distance', 'distance')]
    }

    _default_wrap_angle = 180*u.deg

    def __init__(self, *args, **kwargs):
        wrap = kwargs.pop('wrap_longitude', True)
        super().__init__(*args, **kwargs)
        if wrap and isinstance(self._data, (coord.UnitSphericalRepresentation,
                                            coord.SphericalRepresentation)):
            self._data.lon.wrap_angle = self._default_wrap_angle

    # TODO: remove this. This is a hack required as of astropy v3.1 in order
    # to have the longitude components wrap at the desired angle
    def represent_as(self, base, s='base', in_frame_units=False):
        r = super().represent_as(base, s=s, in_frame_units=in_frame_units)
        if hasattr(r, "lon"):
            r.lon.wrap_angle = self._default_wrap_angle
        return r
    represent_as.__doc__ = coord.BaseCoordinateFrame.represent_as.__doc__


# Rotation matrix defined by trying to align the stream to the equator
R = np.array([[0.94906836, -0.22453560, 0.22102719],
             [-0.06325861, 0.55143610, 0.83181523],
             [-0.30865450, -0.80343138, 0.50914675]])


@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.ICRS,
                                 Pal13Shipp20)
def icrs_to_pal13():
    """ Compute the transformation from Galactic spherical to
        heliocentric Pal 13 coordinates.
    """
    return R


@frame_transform_graph.transform(coord.StaticMatrixTransform, Pal13Shipp20,
                                 coord.ICRS)
def pal13_to_icrs():
    """ Compute the transformation from heliocentric Pal 13 coordinates to
        spherical Galactic.
    """
    return matrix_transpose(icrs_to_pal13())
