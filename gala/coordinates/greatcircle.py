# Built-in
from textwrap import dedent
from warnings import warn

# Third-party
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates.transformations import (DynamicMatrixTransform,
                                                 FunctionTransform)
from astropy.coordinates.attributes import (CoordinateAttribute,
                                            QuantityAttribute)
from astropy.coordinates.matrix_utilities import (rotation_matrix,
                                                  matrix_product,
                                                  matrix_transpose)
from astropy.coordinates.baseframe import base_doc
from astropy.utils.decorators import format_doc
import numpy as np

__all__ = ['GreatCircleICRSFrame', 'make_greatcircle_cls',
           'pole_from_endpoints']


def greatcircle_to_greatcircle(from_greatcircle_coord,
                               to_greatcircle_frame):
    """Transform between two greatcircle frames."""

    # This transform goes through the parent frames on each side.
    # from_frame -> from_frame.origin -> to_frame.origin -> to_frame
    intermediate_from = from_greatcircle_coord.transform_to(
        from_greatcircle_coord.pole)
    intermediate_to = intermediate_from.transform_to(
        to_greatcircle_frame.pole)
    return intermediate_to.transform_to(to_greatcircle_frame)


def reference_to_greatcircle(reference_frame, greatcircle_frame):
    """Convert a reference coordinate to a great circle frame."""

    # Define rotation matrices along the position angle vector, and
    # relative to the origin.
    pole = greatcircle_frame.pole.transform_to(coord.ICRS())
    ra0 = greatcircle_frame.ra0
    center = greatcircle_frame.center
    R_rot = rotation_matrix(greatcircle_frame.rotation, 'z')

    if not np.isnan(ra0) and np.abs(pole.dec.value) > 1e-15:
        zaxis = pole.cartesian.xyz.value
        xaxis = np.array([np.cos(ra0), np.sin(ra0), 0.])
        if np.abs(zaxis[2]) >= 1e-15:
            xaxis[2] = -(zaxis[0]*xaxis[0] + zaxis[1]*xaxis[1]) / zaxis[2]
        else:
            xaxis[2] = 0.
        xaxis = xaxis / np.sqrt(np.sum(xaxis**2))

        yaxis = np.cross(zaxis, xaxis)
        yaxis = yaxis / np.sqrt(np.sum(yaxis**2))

        R = np.stack((xaxis, yaxis, zaxis))

    elif center is not None:
        R1 = rotation_matrix(pole.ra, 'z')
        R2 = rotation_matrix(90*u.deg - pole.dec, 'y')
        Rtmp = matrix_product(R2, R1)

        rot = center.cartesian.transform(Rtmp)
        rot_lon = rot.represent_as(coord.UnitSphericalRepresentation).lon
        R3 = rotation_matrix(rot_lon, 'z')
        R = matrix_product(R3, R2, R1)

    else:
        if not np.isnan(ra0) and np.abs(pole.dec.value) < 1e-15:
            warn("Ignoring input ra0 because the pole is along dec=0",
                 RuntimeWarning)

        R1 = rotation_matrix(pole.ra, 'z')
        R2 = rotation_matrix(90*u.deg - pole.dec, 'y')
        R = matrix_product(R2, R1)

    return matrix_product(R_rot, R)


def greatcircle_to_reference(greatcircle_coord, reference_frame):
    """Convert an great circle frame coordinate to the reference frame"""

    # use the forward transform, but just invert it
    R = reference_to_greatcircle(reference_frame, greatcircle_coord)
    # transpose is the inverse because R is a rotation matrix
    return matrix_transpose(R)


def greatcircle_transforms(self_transform=False):
    def set_greatcircle_transforms(cls):
        DynamicMatrixTransform(reference_to_greatcircle,
                               coord.ICRS, cls,
                               register_graph=coord.frame_transform_graph)

        DynamicMatrixTransform(greatcircle_to_reference,
                               cls, coord.ICRS,
                               register_graph=coord.frame_transform_graph)

        if self_transform:
            FunctionTransform(greatcircle_to_greatcircle,
                              cls, cls,
                              register_graph=coord.frame_transform_graph)
        return cls

    return set_greatcircle_transforms


_components = """
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

_footer = """
Frame attributes
----------------
pole : `~astropy.coordinates.SkyCoord`, `~astropy.coordinates.ICRS`
    The coordinate specifying the pole of this frame.
ra0 : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
    If specified, an additional transformation will be applied to make
    this right ascension the longitude zero-point of the resulting
    coordinate frame.
rotation : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
    If specified, a final rotation about the pole (i.e. the resulting z
    axis) applied.
"""


@format_doc(dedent(base_doc), components=_components, footer=_footer)
@greatcircle_transforms(self_transform=True)
class GreatCircleICRSFrame(coord.BaseCoordinateFrame):
    """A frame rotated into great circle coordinates with the pole and longitude
    specified as frame attributes.

    ``GreatCircleICRSFrame``s always have component names for spherical
    coordinates of ``phi1``/``phi2``.
    """

    pole = CoordinateAttribute(default=None, frame=coord.ICRS)
    center = CoordinateAttribute(default=None, frame=coord.ICRS)
    ra0 = QuantityAttribute(default=np.nan*u.deg, unit=u.deg)
    rotation = QuantityAttribute(default=0, unit=u.deg)

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'phi1'),
            coord.RepresentationMapping('lat', 'phi2'),
            coord.RepresentationMapping('distance', 'distance')]
    }

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    _default_wrap_angle = 180*u.deg

    def __init__(self, *args, **kwargs):
        wrap = kwargs.pop('wrap_longitude', True)
        super().__init__(*args, **kwargs)
        if wrap and isinstance(self._data, (coord.UnitSphericalRepresentation,
                                            coord.SphericalRepresentation)):
            self._data.lon.wrap_angle = self._default_wrap_angle

        if self.center is not None and np.isfinite(self.ra0):
            raise ValueError("Both `center` and `ra0` were specified for this "
                             "{} object: you can only specify one or the other."
                             .format(self.__class__.__name__))

    # TODO: remove this. This is a hack required as of astropy v3.1 in order
    # to have the longitude components wrap at the desired angle
    def represent_as(self, base, s='base', in_frame_units=False):
        r = super().represent_as(base, s=s, in_frame_units=in_frame_units)
        if hasattr(r, "lon"):
            r.lon.wrap_angle = self._default_wrap_angle
        return r
    represent_as.__doc__ = coord.BaseCoordinateFrame.represent_as.__doc__

    @classmethod
    def from_endpoints(cls, coord1, coord2, ra0=None, rotation=None):
        """Compute the great circle frame from two endpoints of an arc on the
        unit sphere.

        Parameters
        ----------
        coord1 : `~astropy.coordinates.SkyCoord`
            One endpoint of the great circle arc.
        coord2 : `~astropy.coordinates.SkyCoord`
            The other endpoint of the great circle arc.
        ra0 : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
            If specified, an additional transformation will be applied to make
            this right ascension the longitude zero-point of the resulting
            coordinate frame.
        rotation : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
            If specified, a final rotation about the pole (i.e. the resulting z
            axis) applied.
        """

        pole = pole_from_endpoints(coord1, coord2)

        kw = dict(pole=pole)
        if ra0 is not None:
            kw['ra0'] = ra0

        if rotation is not None:
            kw['rotation'] = rotation

        if ra0 is None and rotation is None:
            midpt = sph_midpoint(coord1, coord2)
            kw['ra0'] = midpt.ra

        return cls(**kw)

    @classmethod
    def from_xyz(cls, xnew=None, ynew=None, znew=None):
        """Compute the great circle frame from a specification of the coordinate
        axes in the new system.

        Parameters
        ----------
        xnew : astropy ``Representation`` object
            The x-axis in the new system.
        ynew : astropy ``Representation`` object
            The y-axis in the new system.
        znew : astropy ``Representation`` object
            The z-axis in the new system.
        """
        is_none = [xnew is None, ynew is None, znew is None]
        if np.sum(is_none) > 1:
            raise ValueError("At least 2 axes must be specified.")

        if xnew is not None:
            xnew = xnew.to_cartesian()

        if ynew is not None:
            ynew = ynew.to_cartesian()

        if znew is not None:
            znew = znew.to_cartesian()

        if znew is None:
            znew = xnew.cross(ynew)

        if ynew is None:
            ynew = -xnew.cross(znew)

        if xnew is None:
            xnew = ynew.cross(znew)

        pole = coord.SkyCoord(znew, frame='icrs')
        center = coord.SkyCoord(xnew, frame='icrs')
        return cls(pole=pole, center=center)

    @classmethod
    def from_R(cls, R, inverse=False):
        """Compute the great circle frame from a rotation matrix that specifies
        the transformation from ICRS to the new frame.

        Parameters
        ----------
        R : array_like
            The transformation matrix.
        inverse : bool (optional)
            If True, the input rotation matrix is assumed to go from the new
            frame to the ICRS frame..
        """

        if inverse:
            Rinv = R
        else:
            Rinv = np.linalg.inv(R)

        pole = coord.CartesianRepresentation([0, 0, 1.]).transform(Rinv)
        ra0 = coord.CartesianRepresentation([1, 0, 0.]).transform(Rinv)

        pole = coord.SkyCoord(pole, frame='icrs')
        ra0 = ra0.represent_as(coord.SphericalRepresentation)

        return cls(pole=pole, ra0=ra0.lon)


def make_greatcircle_cls(cls_name, docstring_header=None, **kwargs):
    @format_doc(base_doc, components=_components, footer=_footer)
    @greatcircle_transforms(self_transform=False)
    class GCFrame(GreatCircleICRSFrame):
        pole = kwargs.get('pole', None)
        ra0 = kwargs.get('ra0', np.nan*u.deg)
        center = kwargs.get('center', None)
        rotation = kwargs.get('rotation', 0*u.deg)

    GCFrame.__name__ = cls_name
    if docstring_header:
        GCFrame.__doc__ = "{0}\n{1}".format(docstring_header, GCFrame.__doc__)

    return GCFrame


def pole_from_endpoints(coord1, coord2):
    """Compute the pole from a great circle that connects the two specified
    coordinates.

    This assumes a right-handed rule from coord1 to coord2: the pole is the
    north pole under that assumption.

    Parameters
    ----------
    coord1 : `~astropy.coordinates.SkyCoord`
        Coordinate of one point on a great circle.
    coord2 : `~astropy.coordinates.SkyCoord`
        Coordinate of the other point on a great circle.

    Returns
    -------
    pole : `~astropy.coordinates.SkyCoord`
        The coordinates of the pole.
    """
    cart1 = coord1.cartesian.without_differentials()
    cart2 = coord2.cartesian.without_differentials()
    if isinstance(coord1, coord.SkyCoord):
        frame1 = coord1.frame
    elif isinstance(coord1, coord.BaseCoordinateFrame):
        frame1 = coord1
    else:
        raise TypeError('Input coordinate must be a SkyCoord or coordinate frame instance.')

    c1 = cart1 / cart1.norm()

    coord2 = coord2.transform_to(frame1)
    c2 = cart2 / cart2.norm()

    pole = c1.cross(c2)
    pole = pole / pole.norm()
    return frame1.realize_frame(pole)


def sph_midpoint(coord1, coord2):
    """Compute the midpoint between two points on the sphere.

    Parameters
    ----------
    coord1 : `~astropy.coordinates.SkyCoord`
        Coordinate of one point on a great circle.
    coord2 : `~astropy.coordinates.SkyCoord`
        Coordinate of the other point on a great circle.

    Returns
    -------
    midpt : `~astropy.coordinates.SkyCoord`
        The coordinates of the spherical midpoint.
    """
    cart1 = coord1.cartesian.without_differentials()
    cart2 = coord2.cartesian.without_differentials()
    if isinstance(coord1, coord.SkyCoord):
        frame1 = coord1.frame
    elif isinstance(coord1, coord.BaseCoordinateFrame):
        frame1 = coord1
    else:
        raise TypeError('Input coordinate must be a SkyCoord or coordinate frame instance.')

    c1 = cart1 / cart1.norm()

    coord2 = coord2.transform_to(frame1)
    c2 = cart2 / cart2.norm()

    midpt = 0.5 * (c1 + c2)
    usph = midpt.represent_as(coord.UnitSphericalRepresentation)

    return frame1.realize_frame(usph)
