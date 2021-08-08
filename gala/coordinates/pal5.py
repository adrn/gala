""" Astropy coordinate class for the Palomar 5 stream coordinate system """

# Third-party
import numpy as np

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import matrix_transpose

from gala.util import GalaDeprecationWarning

__all__ = ["Pal5PriceWhelan18", "Pal5"]


class Pal5PriceWhelan18(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Pal 5 stream by A. Price-Whelan (2018).

    For more information about this class, see the Astropy documentation
    on coordinate frames in :mod:`~astropy.coordinates`.

    Parameters
    ----------
    representation : :class:`~astropy.coordinates.BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)

    phi1 : angle_like, optional, must be keyword
        The longitude-like angle corresponding to Pal 5's orbit.
    phi2 : angle_like, optional, must be keyword
        The latitude-like angle corresponding to Pal 5's orbit.
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
R = np.array([[-0.65019243, -0.75969758, -0.01045969],
              [-0.62969142, 0.54652698, -0.55208422],
              [0.42513354, -0.3523746, -0.83372274]])

# Extra rotation to put the cluster center at (0, 0)
R2 = np.array([[9.99938314e-01, 1.57847502e-03, -1.09943927e-02],
               [-1.57837962e-03, 9.99998754e-01, 1.73543959e-05],
               [1.09944064e-02, 0.00000000e+00, 9.99939560e-01]])
R = R2 @ R


@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.ICRS,
                                 Pal5PriceWhelan18)
def icrs_to_pal5():
    """ Compute the transformation from Galactic spherical to
        heliocentric Pal 5 coordinates.
    """
    return R


@frame_transform_graph.transform(coord.StaticMatrixTransform, Pal5PriceWhelan18,
                                 coord.ICRS)
def pal5_to_icrs():
    """ Compute the transformation from heliocentric Pal 5 coordinates to
        spherical Galactic.
    """
    return matrix_transpose(icrs_to_pal5())


# TODO: remove this in next version
class Pal5(Pal5PriceWhelan18):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("This frame is deprecated. Use Pal5PriceWhelan18 "
                      "instead.", GalaDeprecationWarning)
        super().__init__(*args, **kwargs)


trans = frame_transform_graph.get_transform(Pal5PriceWhelan18,
                                            coord.ICRS).transforms[0]
frame_transform_graph.add_transform(Pal5, coord.ICRS, trans)
trans = frame_transform_graph.get_transform(coord.ICRS,
                                            Pal5PriceWhelan18).transforms[0]
frame_transform_graph.add_transform(coord.ICRS, Pal5, trans)
