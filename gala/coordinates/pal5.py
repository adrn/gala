""" Astropy coordinate class for the Palomar 5 stream coordinate system """

# Third-party
import numpy as np

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose

__all__ = ["Pal5"]

class Pal5(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Pal 5 stream.

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
            coord.RepresentationMapping('distance', 'distance')]
    }

    _default_wrap_angle = 180*u.deg

    def __init__(self, *args, **kwargs):
        wrap = kwargs.pop('wrap_longitude', True)
        super().__init__(*args, **kwargs)
        if wrap and isinstance(self._data, (coord.UnitSphericalRepresentation,
                                            coord.SphericalRepresentation)):
            self._data.lon.wrap_angle = self._default_wrap_angle

# Rotation matrix defined by trying to align the stream to the equator
R = np.array([[-0.65019243, -0.75969758, -0.01045969],
              [-0.62969142, 0.54652698, -0.55208422],
              [0.42513354, -0.3523746, -0.83372274]])

@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.ICRS, Pal5)
def icrs_to_pal5():
    """ Compute the transformation from Galactic spherical to
        heliocentric Pal 5 coordinates.
    """
    return R

@frame_transform_graph.transform(coord.StaticMatrixTransform, Pal5, coord.ICRS)
def pal5_to_icrs():
    """ Compute the transformation from heliocentric Pal 5 coordinates to
        spherical Galactic.
    """
    return matrix_transpose(icrs_to_pal5())
