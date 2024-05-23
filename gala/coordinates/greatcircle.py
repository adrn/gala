# Built-in
from textwrap import dedent
from warnings import warn

import astropy.coordinates as coord

# Third-party
import astropy.units as u
import numpy as np
from astropy.coordinates.attributes import CoordinateAttribute
from astropy.coordinates.baseframe import base_doc
from astropy.coordinates.transformations import (
    DynamicMatrixTransform,
    FunctionTransform,
)
from astropy.utils.decorators import format_doc

from .helpers import StringValidatedAttribute

__all__ = ["GreatCircleICRSFrame", "make_greatcircle_cls", "pole_from_endpoints"]


def get_xhat(zhat, ra0, tol=1e-10):
    """
    Helper to get the x-hat vector along a great circle defined by the input zhat that
    intersects with the specified longitude (ra0).
    """
    ra0 = 90 * u.deg - ra0

    z1, z2, z3 = zhat

    if np.isclose(z3, 0, atol=tol):  # pole in x-y - can't satisfy ra0
        raise ValueError(
            "Pole is in the x-y plane, so can't satisfy the ra0 requirement"
        )

    denom = (
        z2**2 + z3**2 + 2 * z1 * z2 * np.tan(ra0) + (z1**2 + z3**2) * np.tan(ra0) ** 2
    )
    x1 = -np.tan(ra0) * np.sqrt(z3**2 / denom)
    x2 = x1 / np.tan(ra0)

    if np.isclose(z3, 1, atol=tol):
        x3 = 0.0
    else:
        x3 = (z2 + z1 * np.tan(ra0)) * np.abs(x2) / z3

    return np.array([x1, x2, x3])


def get_origin_from_pole_ra0(pole, ra0, origin_disambiguate=None):
    """
    Figure out the coordinate system origin (i.e. the x-axis, expressed in the old
    coordinate frame). Given just a pole and ra0, there is an ambiguity to the direction
    of the x-axis because the two great circles (defined by pole and ra0) intersect at
    two points. To resolve this ambiguity, you can specify ``origin_disambiguate``,
    which is a coordinate in the old system (ICRS) used to pick the x-axis closest to
    that location. If this is not specified, it uses (RA, Dec)=(0, 0).
    """

    if origin_disambiguate is None:
        origin_disambiguate = coord.SkyCoord(0, 0, unit=u.deg, frame=pole)

    # figure out origin from ra0
    zhat = np.squeeze((pole.cartesian / pole.cartesian.norm()).xyz)
    xhat1 = coord.CartesianRepresentation(get_xhat(zhat, ra0))
    xhat2 = -xhat1

    origin1 = coord.SkyCoord(xhat1, frame=pole, representation_type="unitspherical")
    origin2 = coord.SkyCoord(xhat2, frame=pole, representation_type="unitspherical")

    sep1 = origin_disambiguate.separation(origin1).to_value(u.deg)
    sep2 = origin_disambiguate.separation(origin2).to_value(u.deg)

    # Convention:
    if sep1 <= sep2:
        return origin1
    else:
        return origin2


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
        raise TypeError(
            "Input coordinate must be a SkyCoord or coordinate frame instance."
        )

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
        raise TypeError(
            "Input coordinate must be a SkyCoord or coordinate frame instance."
        )

    c1 = cart1 / cart1.norm()

    coord2 = coord2.transform_to(frame1)
    c2 = cart2 / cart2.norm()

    midpt = 0.5 * (c1 + c2)
    usph = midpt.represent_as(coord.UnitSphericalRepresentation)

    return frame1.realize_frame(usph)


def ensure_orthogonal(pole, origin, priority="origin", tol=1e-10):
    """
    Makes sure the pole and origin are unit vectors, and are orthogonal. Adjusts either
    the pole or origin to make orthogonal if not.

    Parameters
    ----------
    x : array_like
        Must be a unit vector.
    z : array_like
        Must be a unit vector.

    """

    origin = origin.realize_frame(
        origin.represent_as("unitspherical").without_differentials()
    ).squeeze()
    pole = pole.realize_frame(
        pole.represent_as("unitspherical").without_differentials()
    ).squeeze()

    x = np.squeeze(origin.cartesian.xyz)
    z = np.squeeze(pole.cartesian.xyz)
    if np.abs(np.dot(x, z)) > tol:
        if priority == "origin":
            msg = "Keeping the origin fixed and adjusting the pole to be orthogonal."
            z = z - (z @ x) * x
            pole = pole.realize_frame(
                coord.CartesianRepresentation(z), representation_type="unitspherical"
            )

        else:  # validated by class attribute, so assume "pole"
            msg = "Keeping the pole fixed and adjusting the origin to be orthogonal."
            x = x - (x @ z) * z
            origin = origin.realize_frame(
                coord.CartesianRepresentation(x), representation_type="unitspherical"
            )

        warn(
            f"Input origin and pole are not orthogonal. {msg} Use "
            "warnings.simplefilter('ignore') to ignore this warning.",
            RuntimeWarning,
        )

    return pole, origin


def pole_origin_to_R(pole, origin):
    """
    Compute the Cartesian rotation matrix from the given pole and origin.

    This functiona assumes that ``pole`` and ``origin`` are orthogonal.
    """
    if not pole.is_equivalent_frame(origin):
        raise ValueError("The coordinate frame of the input pole and origin must match")

    xaxis = np.squeeze((origin.cartesian / origin.cartesian.norm()).xyz)
    zaxis = np.squeeze((pole.cartesian / pole.cartesian.norm()).xyz)
    yaxis = np.cross(zaxis, xaxis)

    R = np.stack((xaxis, yaxis, zaxis))
    return R


def greatcircle_to_greatcircle(from_greatcircle_coord, to_greatcircle_frame):
    """Transform between two greatcircle frames."""

    # This transform goes through the parent frames on each side.
    # from_frame -> from_frame.origin -> to_frame.origin -> to_frame
    intermediate_from = from_greatcircle_coord.transform_to(from_greatcircle_coord.pole)
    intermediate_to = intermediate_from.transform_to(to_greatcircle_frame.pole)
    return intermediate_to.transform_to(to_greatcircle_frame)


def reference_to_greatcircle(reference_frame, greatcircle_frame):
    """Convert a reference coordinate to a great circle frame."""
    return greatcircle_frame._R


def greatcircle_to_reference(greatcircle_coord, reference_frame):
    """Convert a great circle frame coordinate to the reference frame"""

    # use the forward transform, but just invert it
    R = reference_to_greatcircle(reference_frame, greatcircle_coord)
    # transpose is the inverse because R is a rotation matrix
    return R.T


def greatcircle_transforms(self_transform=False):
    def set_greatcircle_transforms(cls):
        DynamicMatrixTransform(
            reference_to_greatcircle,
            coord.ICRS,
            cls,
            register_graph=coord.frame_transform_graph,
        )

        DynamicMatrixTransform(
            greatcircle_to_reference,
            cls,
            coord.ICRS,
            register_graph=coord.frame_transform_graph,
        )

        if self_transform:
            FunctionTransform(
                greatcircle_to_greatcircle,
                cls,
                cls,
                register_graph=coord.frame_transform_graph,
            )
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
        The pole of the new coordinate frame, defined in the old frame (ICRS).
    origin : `~astropy.coordinates.SkyCoord`, `~astropy.coordinates.ICRS`
        The x-axis (spherical origin) of the new coordinate frame, defined in the old
        frame (ICRS).
"""


@format_doc(dedent(base_doc), components=_components, footer=_footer)
@greatcircle_transforms(self_transform=True)
class GreatCircleICRSFrame(coord.BaseCoordinateFrame):
    """
    A coordinate frame defined by a pole and origin.

    ``GreatCircleICRSFrame``s always have component names for spherical coordinates of
    ``phi1`` and ``phi2`` (so, proper motion components are ``pm_phi1_cosphi2``, etc.).
    """

    pole = CoordinateAttribute(default=None, frame=coord.ICRS)
    origin = CoordinateAttribute(default=None, frame=coord.ICRS)
    priority = StringValidatedAttribute(
        default="origin", valid_values=["origin", "pole"]
    )

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping("lon", "phi1"),
            coord.RepresentationMapping("lat", "phi2"),
            coord.RepresentationMapping("distance", "distance"),
        ]
    }

    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    _default_wrap_angle = 180 * u.deg

    def __init__(self, *args, **kwargs):
        if "ra0" in kwargs:
            raise ValueError(
                "Initializing a GreatCircleICRSFrame with a pole and ra0 is no longer "
                "supported because this does not uniquely determine a coordinate frame."
                " To initialize a frame with a pole and ra0 and ignore the ambiguity, "
                "use the .from_pole_ra0() classmethod."
            )

        if "rotation" in kwargs:
            raise ValueError(
                "Initializing a GreatCircleICRSFrame with a `rotation` is no longer "
                "supported."
            )

        wrap = kwargs.pop("wrap_longitude", True)
        super().__init__(*args, **kwargs)

        if self.pole is None or self.origin is None:
            raise ValueError("You must specify both a pole and an origin")
        pole, origin = ensure_orthogonal(self.pole, self.origin, priority=self.priority)
        self._pole = pole
        self._origin = origin
        self._R = pole_origin_to_R(self.pole, self.origin)

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

    @classmethod
    def from_pole_ra0(cls, pole, ra0, origin_disambiguate=None):
        f"""
        Compute the great circle frame from a pole and RA of longitude=0.

        {get_origin_from_pole_ra0.__doc__}

        Parameters
        ----------
        pole : `~astropy.coordinates.SkyCoord`
            The pole of the new coordinate frame, defined in the old frame (ICRS).
        ra0 : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
            Right Ascension of longitude zero.
        origin_disambiguate :  `~astropy.coordinates.SkyCoord` (optional)
            A sky coordinate in the old frame (ICRS) used to disambiguate the coordinate
            system origin. The x-axis closest to this coordinate is chosen as the new
            system origin / x-axis.
        """
        origin = get_origin_from_pole_ra0(
            pole, ra0, origin_disambiguate=origin_disambiguate
        )
        return cls(pole=pole, origin=origin)

    @classmethod
    def from_endpoints(cls, coord1, coord2, origin=None, ra0=None, priority=None):
        """
        Compute the great circle frame from two endpoints of an arc on the unit sphere.

        If you specify an ``origin``, it should be orthogonal to the pole of the great
        circle defined by ``coord1`` and ``coord2``. If it is not orthogonal to the
        pole, by default, the pole will be adjusted along the great circle connecting
        the pole to the input ``origin``. If you would instead like to keep the pole
        fixed and orthogonalize the ``origin``, pass in ``priority='pole'``.

        Parameters
        ----------
        coord1 : `~astropy.coordinates.SkyCoord`
            One endpoint of the great circle arc.
        coord2 : `~astropy.coordinates.SkyCoord`
            The other endpoint of the great circle arc.
        origin : `~astropy.coordinates.SkyCoord` (optional)
            The x-axis (spherical origin) of the new coordinate frame, defined in the
            old frame (ICRS). This defines the (phi1,phi2)=(0,0)º coordinate.
        ra0 : `~astropy.units.Quantity`, `~astropy.coordinates.Angle` (optional)
            Right Ascension of longitude zero. You can only specify one of ``origin`` or
            ``ra0``.
        priority : str (optional)
            Defines the priority of keeping either the pole or origin fixed when they
            are not orthogonal based on the input.
        """

        if ra0 is not None and origin is not None:
            raise ValueError("You can only pass one of `ra0` or `origin`, not both")

        pole = pole_from_endpoints(coord1.squeeze(), coord2.squeeze())

        if ra0 is not None:
            midpt = sph_midpoint(coord1.squeeze(), coord2.squeeze())
            origin = get_origin_from_pole_ra0(pole, ra0, midpt)
        elif ra0 is None and origin is None:
            origin = sph_midpoint(coord1.squeeze(), coord2.squeeze())

        return cls(pole=pole, origin=origin, priority=priority)

    @classmethod
    def from_xyz(cls, xnew=None, ynew=None, znew=None):
        """
        Compute the great circle frame from a specification of the coordinate axes in
        the new system.

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

        pole = coord.SkyCoord(znew, frame="icrs")
        origin = coord.SkyCoord(xnew, frame="icrs")
        return cls(pole=pole, origin=origin)

    @classmethod
    def from_R(cls, R):
        """
        Compute the great circle frame from a rotation matrix that specifies the
        transformation from ICRS to the new frame.

        Parameters
        ----------
        R : array_like
            The transformation matrix.
        """

        pole = coord.SkyCoord(
            coord.CartesianRepresentation([0.0, 0.0, 1.0]).transform(R.T),
            frame="icrs",
            representation_type="unitspherical",
        )
        origin = coord.SkyCoord(
            coord.CartesianRepresentation([1.0, 0.0, 0.0]).transform(R.T),
            frame="icrs",
            representation_type="unitspherical",
        )

        return cls(pole=pole, origin=origin)


def make_greatcircle_cls(cls_name, docstring_header=None, **kwargs):
    @format_doc(base_doc, components=_components, footer=_footer)
    @greatcircle_transforms(self_transform=False)
    class GCFrame(GreatCircleICRSFrame):
        pole = CoordinateAttribute(default=kwargs.get("pole", None), frame=coord.ICRS)
        origin = CoordinateAttribute(
            default=kwargs.get("origin", None), frame=coord.ICRS
        )

    GCFrame.__name__ = cls_name
    if docstring_header:
        GCFrame.__doc__ = "{0}\n{1}".format(docstring_header, GCFrame.__doc__)

    return GCFrame
