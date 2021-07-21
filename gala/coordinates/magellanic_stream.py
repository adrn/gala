""" Astropy coordinate class for the Magellanic Stream coordinate system """

from astropy.coordinates.matrix_utilities import (rotation_matrix,
                                                  matrix_product,
                                                  matrix_transpose)
from astropy.coordinates.baseframe import (frame_transform_graph,
                                           BaseCoordinateFrame,
                                           RepresentationMapping)
from astropy.coordinates.transformations import StaticMatrixTransform
from astropy.coordinates import representation as r
from astropy.coordinates import Galactic

import astropy.units as u

from gala.util import GalaDeprecationWarning

__all__ = ["MagellanicStreamNidever08", "MagellanicStream"]


class MagellanicStreamNidever08(BaseCoordinateFrame):
    """
    A coordinate or frame aligned with the Magellanic Stream,
    as defined by Nidever et al. (2008,
    see: `<http://adsabs.harvard.edu/abs/2008ApJ...679..432N>`_).

    For more information about this class, see the Astropy documentation
    on coordinate frames in :mod:`~astropy.coordinates`.

    Examples
    --------
    Converting the coordinates of the Large Magellanic Cloud:

        >>> from astropy import coordinates as coord
        >>> from astropy import units as u
        >>> from gala.coordinates import MagellanicStreamNidever08

        >>> c = coord.Galactic(l=280.4652*u.deg, b=-32.8884*u.deg)
        >>> ms = c.transform_to(MagellanicStreamNidever08())
        >>> print(ms)
        <MagellanicStreamNidever08 Coordinate: (L, B) in deg
            (-0.13686116, 2.42583948)>
    """

    frame_specific_representation_info = {
        r.SphericalRepresentation: [
            RepresentationMapping('lon', 'L'),
            RepresentationMapping('lat', 'B')
        ]
    }

    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential

    _ngp = Galactic(l=188.5*u.deg, b=-7.5*u.deg)
    _lon0 = Galactic(l=280.47*u.deg, b=-32.75*u.deg)

    _default_wrap_angle = 180*u.deg

    def __init__(self, *args, **kwargs):
        wrap = kwargs.pop('wrap_longitude', True)
        super().__init__(*args, **kwargs)
        if wrap and isinstance(self._data, (r.UnitSphericalRepresentation,
                                            r.SphericalRepresentation)):
            self._data.lon.wrap_angle = self._default_wrap_angle

    # TODO: remove this. This is a hack required as of astropy v3.1 in order
    # to have the longitude components wrap at the desired angle
    def represent_as(self, base, s='base', in_frame_units=False):
        r = super().represent_as(base, s=s, in_frame_units=in_frame_units)
        if hasattr(r, "lon"):
            r.lon.wrap_angle = self._default_wrap_angle
        return r
    represent_as.__doc__ = BaseCoordinateFrame.represent_as.__doc__


@frame_transform_graph.transform(StaticMatrixTransform,
                                 Galactic, MagellanicStreamNidever08)
def gal_to_mag():
    mat1 = rotation_matrix(57.275785782128686*u.deg, 'z')
    mat2 = rotation_matrix(90*u.deg - MagellanicStreamNidever08._ngp.b, 'y')
    mat3 = rotation_matrix(MagellanicStreamNidever08._ngp.l, 'z')

    return matrix_product(mat1, mat2, mat3)


@frame_transform_graph.transform(StaticMatrixTransform,
                                 MagellanicStreamNidever08, Galactic)
def mag_to_gal():
    return matrix_transpose(gal_to_mag())


# TODO: remove this in next version
class MagellanicStream(MagellanicStreamNidever08):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn("This frame is deprecated. Use MagellanicStreamNidever08 "
                      "instead.", GalaDeprecationWarning)
        super().__init__(*args, **kwargs)


trans = frame_transform_graph.get_transform(MagellanicStreamNidever08,
                                            Galactic).transforms[0]
frame_transform_graph.add_transform(MagellanicStream, Galactic, trans)
trans = frame_transform_graph.get_transform(Galactic,
                                            MagellanicStreamNidever08).transforms[0]
frame_transform_graph.add_transform(Galactic, MagellanicStream, trans)
