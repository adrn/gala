# coding: utf-8

""" Transform a proper motion to/from Galactic to/from ICRS coordinates """

from __future__ import division, print_function

# Standard library
import warnings

# Third-party
import numpy as np
from astropy.utils.misc import isiterable
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph
try:
    from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
    ASTROPY_1_3 = True
except ImportError:
    from .matrix_utilities import rotation_matrix, matrix_product, matrix_transpose
    ASTROPY_1_3 = False

if not ASTROPY_1_3:
    import astropy
    import warnings
    warnings.warn("We recommend using Astropy v1.3 or later. You have: {}"
                  .format(astropy.__version__), DeprecationWarning)

__all__ = ['transform_proper_motion', 'pm_gal_to_icrs', 'pm_icrs_to_gal']

def normal_triad(coordinate):
    """
    Returns the "normal triad" at the specified sky-coordinate. Briefly, the
    normal triad is a set of (unit) basis vectors on the tangent plane at the
    coordinate that specify (1) the direction to the point, (2) the direction
    of increasing longitude, and (3) the direction of increasing latitude.

    See Section 4.3 (especially 4.3.3) in "Astrometry for Astrophysics:
    Methods, Models, and Applications".

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
        An instance of an Astropy coordinate object.

    Returns
    -------
    r_hat : `numpy.ndarray`
    p_hat : `numpy.ndarray`
    q_hat : `numpy.ndarray`

    """

    sin_lat,cos_lat = np.sin(coordinate.spherical.lat), np.cos(coordinate.spherical.lat)
    sin_lon,cos_lon = np.sin(coordinate.spherical.lon), np.cos(coordinate.spherical.lon)

    r_hat = np.stack((cos_lat*cos_lon, cos_lat*sin_lon, sin_lat))
    p_hat = np.stack((-sin_lon, cos_lon, np.zeros_like(sin_lon)))
    q_hat = np.stack((-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat))

    return r_hat, p_hat, q_hat

def _get_rotation(coordinate, R):
    """
    Get the tangent-plane rotation matrix components to rotate the
    proper-motion vector to the new coordinate frame (specified by the
    full-space rotation matrix ``R``)
    """
    r, p, q = normal_triad(coordinate)

    tang_rot = np.cross(R[2], r.T)
    tang_rot = tang_rot / np.linalg.norm(tang_rot, axis=-1)[...,None]

    return np.sum(tang_rot * p.T, axis=-1).T, np.sum(tang_rot * q.T, axis=-1).T

def rotate_proper_motion(coordinate, pm, rot_matrix):
    """
    Rotate the proper motion vector ``pm`` at the sky coordinate
    ``coordinate`` using the full-space rotation matrix ``rot_matrix``
    that defines the coordinate transformation from the frame of the
    input coordinates to the desired coordinate system.

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
        An instance of an Astropy coordinate object.
    pm : :class:`~astropy.units.Quantity`, iterable
        Proper motion components (longitude, latitude) in the same frame
        as the input coordinates. Can either be an iterable of two
        :class:`~astropy.units.Quantity` objects or a single
        :class:`~astropy.units.Quantity` with shape (2,N).
        The proper motion in longitude is assumed to already include
        the cos(latitude) term.
    rot_matrix : `numpy.ndarray`
        A 3 by 3 rotation matrix that transforms from the input coordinate
        frame to the desired frame.

    Returns
    -------
    new_pm : tuple
        A length-2 tuple containing the proper motion components
        (longitude, latitude) in the new frame. The longitude component is
        includes the cos(latitude) term.

    """
    c, s = _get_rotation(coordinate, rot_matrix)
    return c*pm[0] + s*pm[1], c*pm[1] - s*pm[0]

def get_full_transformation_matrix(fromcoord, toframe):
    """
    This is ripped off of the ``CompositeTransform`` class in
    astropy.coordinates
    """

    if isinstance(fromcoord, coord.SkyCoord):
        fromcoord = fromcoord.frame

    composite_trans = frame_transform_graph.get_transform(fromcoord.__class__, toframe)

    full_matrix = None
    curr_coord = fromcoord
    for t in composite_trans.transforms:
        # build an intermediate frame with attributes taken from either
        # `fromframe`, or if not there, `toframe`, or if not there, use
        # the defaults
        frattrs = {}
        for inter_frame_attr_nm in t.tosys.get_frame_attr_names():
            if hasattr(toframe, inter_frame_attr_nm):
                attr = getattr(toframe, inter_frame_attr_nm)
                frattrs[inter_frame_attr_nm] = attr

            elif hasattr(fromcoord, inter_frame_attr_nm):
                attr = getattr(fromcoord, inter_frame_attr_nm)
                frattrs[inter_frame_attr_nm] = attr

        curr_toframe = t.tosys(**frattrs)

        if hasattr(t, 'matrix'):
            if full_matrix is None:
                full_matrix = t.matrix
            else:
                full_matrix = matrix_product(full_matrix, t.matrix)

        elif hasattr(t, 'matrix_func'):
            _matrix = t.matrix_func(curr_coord, curr_toframe)
            if full_matrix is None:
                full_matrix = _matrix
            else:
                full_matrix = matrix_product(full_matrix, _matrix)

        else:
            raise ValueError("To get the transformation matrix, all intermediate "
                             "transformations must use matrices ({} to {} does not)."
                             .format(curr_coord, t.tosys))
        curr_coord = t(curr_coord, curr_toframe)

    if full_matrix is None:
        full_matrix = np.eye(3)

    return full_matrix

def transform_proper_motion(coordinate, pm, new_frame):
    """
    Transform the proper motion vector ``pm`` at the sky coordinate
    ``coordinate`` to the desired coordinate frame ``new_frame``.

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
        An instance of an Astropy coordinate object.
    pm : :class:`~astropy.units.Quantity`, iterable
        Proper motion components (longitude, latitude) in the same frame
        as the input coordinates. Can either be an iterable of two
        :class:`~astropy.units.Quantity` objects or a single
        :class:`~astropy.units.Quantity` with shape (2,N).
        The proper motion in longitude is assumed to already include
        the cos(latitude) term.
    new_frame : :class:`~astropy.coordinates.BaseCoordinateFrame` subclass
        The desired coordinate frame of the proper motion.

    Returns
    -------
    new_pm : tuple
        A length-2 tuple containing the proper motion components
        (longitude, latitude) in the new frame. The longitude component is
        includes the cos(latitude) term.

    Examples
    --------

        >>> import astropy.units as u
        >>> import astropy.coordinates as coord
        >>> c = coord.SkyCoord(ra=196.5*u.degree, dec=-10.33*u.deg, distance=16.2*u.kpc)
        >>> pm = [-1.53, 3.5]*u.mas/u.yr
        >>> transform_proper_motion(c, pm, coord.Galactic) # doctest: +FLOAT_CMP
        <Quantity [-1.19944367, 3.62660101] mas / yr>

    """
    R = get_full_transformation_matrix(coordinate, new_frame)
    new_pm = rotate_proper_motion(coordinate, pm, R)
    new_pm = np.stack((new_pm[0].value, new_pm[1].to(new_pm[0].unit))).value * new_pm[0].unit

    return new_pm

# ----------------------------------------------------------------------------
# Deprecated:
#

def pm_gal_to_icrs(coordinate, pm):
    r"""
    Convert proper motion in Galactic coordinates (l,b) to
    ICRS coordinates (RA, Dec).

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
        An instance of an Astropy coordinate object. Can be in any
        frame that is transformable to ICRS coordinates.
    pm : :class:`~astropy.units.Quantity`, iterable
        Full description of proper motion in Galactic longitude and
        latitude. Can either be a tuple of two
        :class:`~astropy.units.Quantity` objects or a single
        :class:`~astropy.units.Quantity` with shape (2,N).
        The proper motion in longitude is assumed to be multipled by
        cosine of Galactic latitude, :math:`\mu_l\cos b`.

    Returns
    -------
    pm : :class:`~astropy.units.Quantity`
        An astropy :class:`~astropy.units.Quantity` object specifying the
        proper motion vector array in ICRS coordinates. Will have shape
        (2,N).

    Examples
    --------

        >>> import astropy.units as u
        >>> import astropy.coordinates as coord
        >>> c = coord.SkyCoord(ra=196.5*u.degree, dec=-10.33*u.deg, distance=16.2*u.kpc)
        >>> pm = [-1.53, 3.5]*u.mas/u.yr
        >>> pm_gal_to_icrs(c, pm) # doctest: +FLOAT_CMP
        <Quantity [-1.84741767, 3.34334366] mas / yr>

    """

    warnings.warn("This function is now deprecated. Use "
                  "`gala.coordinates.transform_proper_motion()` instead. The proper motion "
                  "transformations now support a much wider range of input and output frames "
                  "and are more accurate.", DeprecationWarning)

    g = coordinate.transform_to(coord.Galactic)
    i = coordinate.transform_to(coord.ICRS)

    mulcosb,mub = map(np.atleast_1d, pm)
    shape = mulcosb.shape

    # coordinates of NGP
    ag = coord.Galactic._ngp_J2000.ra
    dg = coord.Galactic._ngp_J2000.dec

    # used for the transformation matrix from Bovy (2011)
    cosphi = (np.sin(dg) - np.sin(i.dec)*np.sin(g.b)) / np.cos(g.b) / np.cos(i.dec)
    sinphi = np.sin(i.ra - ag) * np.cos(dg) / np.cos(g.b)

    R = np.zeros((2,2)+shape)
    R[0,0] = cosphi
    R[0,1] = -sinphi
    R[1,0] = sinphi
    R[1,1] = cosphi

    mu = np.vstack((mulcosb.value[None], mub.to(mulcosb.unit).value[None]))
    # new_mu = np.zeros_like(mu)
    # for i in range(n):
    #     new_mu[:,i] = R[...,i].dot(mu[:,i])
    new_mu = np.einsum('ijk...,jk...->ik...',R,mu)

    if coordinate.isscalar:
        return new_mu.reshape((2,))*mulcosb.unit
    else:
        return new_mu.reshape((2,) + mulcosb.shape)*mulcosb.unit

def pm_icrs_to_gal(coordinate, pm):
    r"""
    Convert proper motion in ICRS coordinates (RA, Dec) to
    Galactic coordinates (l,b).

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
        An instance of an Astropy coordinate object. Can be in any
        frame that is transformable to ICRS coordinates.
    pm : :class:`~astropy.units.Quantity`, iterable
        Full description of proper motion in Right ascension (RA) and
        declination (Dec). Can either be a tuple of two
        :class:`~astropy.units.Quantity` objects or a single
        :class:`~astropy.units.Quantity` with shape (2,N).
        The proper motion in RA is assumed to be multipled by
        cosine of declination, :math:`\mu_\alpha\cos\delta`.

    Returns
    -------
    pm : :class:`~astropy.units.Quantity`
        An astropy :class:`~astropy.units.Quantity` object specifying the
        proper motion vector array in Galactic coordinates. Will have shape
        (2,N).

    Examples
    --------

        >>> import astropy.units as u
        >>> import astropy.coordinates as coord
        >>> c = coord.SkyCoord(ra=196.5*u.degree, dec=-10.33*u.deg, distance=16.2*u.kpc)
        >>> pm = [-1.84741767, 3.34334366]*u.mas/u.yr
        >>> pm_icrs_to_gal(c, pm) # doctest: +FLOAT_CMP
        <Quantity [-1.52999988, 3.49999973] mas / yr>

    """

    warnings.warn("This function is now deprecated. Use "
                  "`gala.coordinates.transform_proper_motion()` instead. The proper motion "
                  "transformations now support a much wider range of input and output frames "
                  "and are more accurate.", DeprecationWarning)

    g = coordinate.transform_to(coord.Galactic)
    i = coordinate.transform_to(coord.ICRS)

    muacosd,mud = map(np.atleast_1d, pm)
    shape = muacosd.shape

    # coordinates of NGP
    ag = coord.Galactic._ngp_J2000.ra
    dg = coord.Galactic._ngp_J2000.dec

    # used for the transformation matrix from Bovy (2011)
    cosphi = (np.sin(dg) - np.sin(i.dec)*np.sin(g.b)) / np.cos(g.b) / np.cos(i.dec)
    sinphi = np.sin(i.ra - ag) * np.cos(dg) / np.cos(g.b)

    R = np.zeros((2,2)+shape)
    R[0,0] = cosphi
    R[0,1] = sinphi
    R[1,0] = -sinphi
    R[1,1] = cosphi

    mu = np.vstack((muacosd.value[None], mud.to(muacosd.unit).value[None]))
    # new_mu = np.zeros_like(mu)
    # for i in range(n):
    #     new_mu[:,i] = R[...,i].dot(mu[:,i])
    new_mu = np.einsum('ijk...,jk...->ik...',R,mu)

    if coordinate.isscalar:
        return new_mu.reshape((2,))*muacosd.unit
    else:
        return new_mu.reshape((2,) + muacosd.shape)*muacosd.unit
