""" Astropy coordinate class for the Sagittarius coordinate system """

from astropy.coordinates import frame_transform_graph
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose

__all__ = ["Orphan"]

class Orphan(coord.BaseCoordinateFrame):
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

# Define the Euler angles
phi = 128.79 * u.degree
theta = 54.39 * u.degree
psi = 90.70 * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(phi, "z")
C = rotation_matrix(theta, "x")
B = rotation_matrix(psi, "z")
R = matrix_product(B, C, D)

@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Orphan)
def galactic_to_orp():
    """ Compute the transformation from Galactic spherical to
        heliocentric Orphan coordinates.
    """
    return R

# Oph to Galactic coordinates
@frame_transform_graph.transform(coord.StaticMatrixTransform, Orphan, coord.Galactic)
def oph_to_galactic():
    """ Compute the transformation from heliocentric Orphan coordinates to
        spherical Galactic.
    """
    return matrix_transpose(galactic_to_orp())
