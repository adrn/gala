# coding: utf-8

""" Miscellaneous astronomical velocity transformations. """

from __future__ import division, print_function


# Third-party
import numpy as np
from numpy import cos, sin

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.builtin_frames.galactocentric import _ROLL0 as ROLL0
try:
    from astropy.coordinates.matrix_utilities import (rotation_matrix,
                                                      matrix_product)
    ASTROPY_1_3 = True
except ImportError:
    from .matrix_utilities import rotation_matrix, matrix_product
    ASTROPY_1_3 = False

if not ASTROPY_1_3:
    import astropy
    import warnings
    warnings.warn("We recommend using Astropy v1.3 or later. You have: {}"
                  .format(astropy.__version__), DeprecationWarning)

# Package
from .propermotion import transform_proper_motion

__all__ = ["vgal_to_hel", "vhel_to_gal", "vgsr_to_vhel", "vhel_to_vgsr"]

# This is the default circular velocity and LSR peculiar velocity of the Sun
# TODO: make this a config item?
VCIRC = 220. * u.km/u.s
VLSR = [10., 5.25, 7.17] * u.km/u.s

def _icrs_gctc_velocity_matrix(galactocentric_frame):
    """
    Construct a transformation matrix to go from heliocentric ICRS to a galactocentric
    frame. This is just a rotation and tilt which makes it approximately the same
    as transforming to Galactic coordinates. This only works for velocity because there
    is no shift due to the position of the Sun.
    """

    # define rotation matrix to align x(ICRS) with the vector to the Galactic center
    M1 = rotation_matrix(-galactocentric_frame.galcen_dec, 'y')
    M2 = rotation_matrix(galactocentric_frame.galcen_ra, 'z')

    # extra roll away from the Galactic x-z plane
    M3 = rotation_matrix(ROLL0 - galactocentric_frame.roll, 'x')

    # rotate about y' to account for tilt due to Sun's height above the plane
    z_d = (galactocentric_frame.z_sun / galactocentric_frame.galcen_distance).decompose()
    M4 = rotation_matrix(-np.arcsin(z_d), 'y')

    return matrix_product(M4, M3, M1, M2)  # this is right: 4,3,1,2

def vgal_to_hel(coordinate, velocity, vcirc=None, vlsr=None,
                galactocentric_frame=None):
    r"""
    Convert a Galactocentric velocity to a Heliocentric velocity.

    The frame of the input coordinate determines the output frame of the
    heliocentric velocity. For example, if the input coordinate is in the ICRS
    frame, heliocentric velocity will also be in the ICRS.

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
        This is most commonly a :class:`~astropy.coordinates.SkyCoord` object,
        but alternatively, it can be any coordinate frame object that is
        transformable to the Galactocentric frame.
    velocity : :class:`~astropy.coordinates.BaseDifferential`, :class:`~astropy.units.Quantity`, iterable
        If not provided as a Differential instance, the velocity components are
        assumed to be Cartesian :math:`(v_x,v_y,v_z)` and should either
        be a single :class:`~astropy.units.Quantity` object with shape ``(3,N)``
        or an iterable object with 3 :class:`~astropy.units.Quantity` objects as
        elements.
    vcirc : :class:`~astropy.units.Quantity` (optional)
        Circular velocity of the Sun.
    vlsr : :class:`~astropy.units.Quantity` (optional)
        Velocity of the Sun relative to the local standard
        of rest (LSR).
    galactocentric_frame : :class:`~astropy.coordinates.Galactocentric` (optional)
        An instantiated :class:`~astropy.coordinates.Galactocentric` frame
        object with custom parameters for the Galactocentric coordinates. For
        example, if you want to set your own position of the Galactic center,
        you can pass in a frame with custom `galcen_ra` and `galcen_dec`.

    Returns
    -------
    helio_velocity : tuple
        The computed heliocentric velocity.

    Examples
    --------

        >>> import astropy.units as u
        >>> import astropy.coordinates as coord
        >>> c = coord.Galactocentric([15., 13, 2]*u.kpc)
        >>> vxyz = [-115., 100., 95.]*u.km/u.s
        >>> icrs = c.transform_to(coord.ICRS)
        >>> vgal_to_hel(icrs, vxyz) # doctest: +FLOAT_CMP
        <SphericalDifferential (d_lon, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
            (-0.87712464,  0.02450121, -163.24449462)>

        >>> c = coord.Galactocentric([[15.,11.], [13,21.], [2.,-7]]*u.kpc)
        >>> vxyz = [[-115.,11.], [100.,-21.], [95.,103]] * u.km/u.s
        >>> icrs = c.transform_to(coord.ICRS)
        >>> vhel = vgal_to_hel(icrs, vxyz)
        >>> vhel # doctest: +FLOAT_CMP
        <SphericalDifferential (d_lon, d_lat, d_distance) in (mas / yr, mas / yr, km / s)
            [(-0.87712464,  0.02450121, -163.24449462),
             (-0.91690268, -0.86124895, -198.31241148)]>
        >>> vhel.d_lon # doctest: +FLOAT_CMP
        <Quantity [-0.87712464,-0.91690268] mas / yr>

    """

    if vcirc is None:
        vcirc = VCIRC

    if vlsr is None:
        vlsr = VLSR

    if galactocentric_frame is None:
        galactocentric_frame = coord.Galactocentric

    c = coordinate
    v = velocity
    if isinstance(v, coord.BaseDifferential):
        pos = c.represent_as(v.base_representation)
        vxyz = v.represent_as(coord.CartesianDifferential, base=pos).d_xyz

    else:
        vxyz = v

    R = _icrs_gctc_velocity_matrix(galactocentric_frame)

    # remove circular and LSR velocities
    vxyz[1] = vxyz[1] - vcirc
    for i in range(3):
        vxyz[i] = vxyz[i] - vlsr[i]

    orig_shape = vxyz.shape
    # v_icrs = np.linalg.inv(R).dot(vxyz.reshape(vxyz.shape[0], np.prod(vxyz.shape[1:]))).reshape(orig_shape)
    v_icrs = R.T.dot(vxyz.reshape(vxyz.shape[0], -1)).reshape(orig_shape)

    # get cartesian heliocentric
    x_icrs = c.transform_to(coord.ICRS).cartesian.xyz
    d = np.sqrt(np.sum(x_icrs**2, axis=0))
    dxy = np.sqrt(x_icrs[0]**2 + x_icrs[1]**2)

    vr = np.sum(x_icrs * v_icrs, axis=0) / d

    mua = ((x_icrs[0]*v_icrs[1] - v_icrs[0]*x_icrs[1]) / dxy**2)
    mua_cosd = (mua * dxy / d).to(u.mas/u.yr,
                                  equivalencies=u.dimensionless_angles())

    mud = (-(x_icrs[2]*(x_icrs[0]*v_icrs[0] + x_icrs[1]*v_icrs[1]) -
             dxy**2*v_icrs[2]) / d**2 / dxy)
    mud = mud.to(u.mas/u.yr, equivalencies=u.dimensionless_angles())

    pm_radec = (mua_cosd, mud)

    if c.name == 'icrs':
        pm = u.Quantity(map(np.atleast_1d, pm_radec))

    else:
        # transform ICRS proper motions to whatever frame
        pm = transform_proper_motion(c.transform_to(coord.ICRS), pm_radec, c)

    if c.isscalar:
        vr = vr.reshape(())
        pm = (pm[0].reshape(()), pm[1].reshape(()))

    # NOTE: remember that d_lon does not include the cos(lat) term!
    return coord.SphericalDifferential(d_lon=pm[0] / np.cos(c.spherical.lat),
                                       d_lat=pm[1],
                                       d_distance=vr)

def vhel_to_gal(coordinate, velocity, vcirc=None, vlsr=None,
                galactocentric_frame=None):
    r"""
    Convert a Heliocentric velocity to a Galactocentric velocity.

    The frame of the input coordinate determines how to interpret the given
    proper motions. For example, if the input coordinate is in the ICRS frame,
    the input velocity is assumed to be as well.

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
        This is most commonly a :class:`~astropy.coordinates.SkyCoord` object,
        but alternatively, it can be any coordinate frame object that is
        transformable to the Galactocentric frame.
    velocity : :class:`~astropy.coordinates.BaseDifferential`, iterable
        If not provided as a Differential instance, the velocity input is
        assumed to be a length-3 iterable containing proper motion components
        and radial velocity. The proper motion in longitude should *not* contain
        the cos(latitude) term.
    vcirc : :class:`~astropy.units.Quantity` (optional)
        Circular velocity of the Sun.
    vlsr : :class:`~astropy.units.Quantity` (optional)
        Velocity of the Sun relative to the local standard
        of rest (LSR).
    galactocentric_frame : :class:`~astropy.coordinates.Galactocentric` (optional)
        An instantiated :class:`~astropy.coordinates.Galactocentric` frame
        object with custom parameters for the Galactocentric coordinates. For
        example, if you want to set your own position of the Galactic center,
        you can pass in a frame with custom `galcen_ra` and `galcen_dec`.

    Returns
    -------
    vxyz : :class:`~astropy.units.Quantity` (optional)
        Cartesian velocity components (U,V,W). A :class:`~astropy.units.Quantity`
        object with shape (3,N).

    Examples
    --------

        >>> import astropy.units as u
        >>> import astropy.coordinates as coord
        >>> c = coord.SkyCoord(ra=196.5*u.degree, dec=-10.33*u.deg,
        ...                    distance=16.2*u.kpc)
        >>> vhel = coord.SphericalDifferential(d_lon=-1.53*u.mas/u.yr,
        ...                                    d_lat=3.5*u.mas/u.yr,
        ...                                    d_distance=161.4*u.km/u.s)
        >>> vgal = vhel_to_gal(c, vhel)
        >>> vgal # doctest: +FLOAT_CMP
        <CartesianDifferential (d_x, d_y, d_z) in km / s
            (-135.73494236,  263.72006872,  305.39515348)>

        >>> c = coord.SkyCoord(ra=[196.5,51.3]*u.degree, dec=[-10.33,2.1]*u.deg,
        ...                    distance=[16.2,11.]*u.kpc)
        >>> vhel = coord.SphericalDifferential(d_lon=[-1.53, 4.5]*u.mas/u.yr,
        ...                                    d_lat=[3.5, 10.9]*u.mas/u.yr,
        ...                                    d_distance=[161.4, -210.2]*u.km/u.s)
        >>> vgal = vhel_to_gal(c, vhel)
        >>> vgal # doctest: +FLOAT_CMP
        <CartesianDifferential (d_x, d_y, d_z) in km / s
            [(-135.73494236,  263.72006872,  305.39515348),
             (-212.0251261 ,  496.96148064,  554.07817075)]>

    """

    if vcirc is None:
        vcirc = VCIRC

    if vlsr is None:
        vlsr = VLSR

    if galactocentric_frame is None:
        galactocentric_frame = coord.Galactocentric

    c = coordinate
    v = velocity
    if isinstance(v, coord.BaseDifferential):
        pos = c.represent_as(v.base_representation)
        vsph = v.represent_as(coord.SphericalDifferential, base=pos)
        vsph = [vsph.d_lon, vsph.d_lat, vsph.d_distance]

    else:
        vsph = v

    pm = vsph[:2]
    rv = vsph[2]

    if c.name == 'icrs':
        pm_radec = u.Quantity([np.atleast_1d(pm[0]) * np.cos(c.dec),
                               np.atleast_1d(pm[1])])
        icrs = c

    else:
        pm[0] = pm[0] * np.cos(c.spherical.lat)
        # pmra *includes* cos(dec) term!
        pm_radec = transform_proper_motion(c, pm, coord.ICRS)
        icrs = c.transform_to(coord.ICRS)

    # I'm so fired
    a,d,D = icrs.ra, icrs.dec, c.distance

    # proper motion components: longitude, latitude
    mura_cosdec, mudec = pm_radec
    vra = (D*mura_cosdec).to(rv.unit, equivalencies=u.dimensionless_angles())
    vdec = (D*mudec).to(rv.unit, equivalencies=u.dimensionless_angles())

    v_icrs = [rv*np.cos(a)*np.cos(d) - vra*np.sin(a) - vdec*np.cos(a)*np.sin(d),
              rv*np.sin(a)*np.cos(d) + vra*np.cos(a) - vdec*np.sin(a)*np.sin(d),
              rv*np.sin(d) + vdec*np.cos(d)]
    v_icrs = np.array([v.to(u.km/u.s).value for v in v_icrs]) * u.km/u.s

    R = _icrs_gctc_velocity_matrix(galactocentric_frame)

    orig_shape = v_icrs.shape
    v_gc = R.dot(v_icrs.reshape(v_icrs.shape[0], -1)).reshape(orig_shape)

    # remove circular and LSR velocities
    v_gc[1] = v_gc[1] + vcirc
    for i in range(3):
        v_gc[i] = v_gc[i] + vlsr[i]

    if c.isscalar:
        v_gc = v_gc.reshape((3,))

    return coord.CartesianDifferential(*v_gc)

# -----------------------------------------------------------------------------

def vgsr_to_vhel(coordinate, vgsr, vcirc=None, vlsr=None):
    """
    Convert a radial velocity in the Galactic standard of rest (GSR) to
    a barycentric radial velocity.

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`
        An Astropy SkyCoord object or anything object that can be passed
        to the SkyCoord initializer.
    vgsr : :class:`~astropy.units.Quantity`
        GSR line-of-sight velocity.
    vcirc : :class:`~astropy.units.Quantity`
        Circular velocity of the Sun.
    vlsr : :class:`~astropy.units.Quantity`
        Velocity of the Sun relative to the local standard
        of rest (LSR).

    Returns
    -------
    vhel : :class:`~astropy.units.Quantity`
        Radial velocity in a barycentric rest frame.

    """

    if vcirc is None:
        vcirc = VCIRC

    if vlsr is None:
        vlsr = VLSR

    c = coord.SkyCoord(coordinate)
    g = c.galactic
    l,b = g.l, g.b

    if not isinstance(vgsr, u.Quantity):
        raise TypeError("vgsr must be a Quantity subclass")

    # compute the velocity relative to the LSR
    lsr = vgsr - vcirc*sin(l)*cos(b)

    # velocity correction for Sun relative to LSR
    v_correct = vlsr[0]*cos(b)*cos(l) + \
        vlsr[1]*cos(b)*sin(l) + \
        vlsr[2]*sin(b)
    vhel = lsr - v_correct

    return vhel

def vhel_to_vgsr(coordinate, vhel, vcirc=None, vlsr=None):
    """
    Convert a velocity from a heliocentric radial velocity to
    the Galactic standard of rest (GSR).

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`
        An Astropy SkyCoord object or anything object that can be passed
        to the SkyCoord initializer.
    vhel : :class:`~astropy.units.Quantity`
        Barycentric line-of-sight velocity.
    vcirc : :class:`~astropy.units.Quantity`
        Circular velocity of the Sun.
    vlsr : :class:`~astropy.units.Quantity`
        Velocity of the Sun relative to the local standard
        of rest (LSR).

    Returns
    -------
    vgsr : :class:`~astropy.units.Quantity`
        Radial velocity in a galactocentric rest frame.

    """

    if vcirc is None:
        vcirc = VCIRC

    if vlsr is None:
        vlsr = VLSR

    c = coord.SkyCoord(coordinate)
    g = c.galactic
    l,b = g.l, g.b

    if not isinstance(vhel, u.Quantity):
        raise TypeError("vhel must be a Quantity subclass")

    lsr = vhel + vcirc*sin(l)*cos(b)

    # velocity correction for Sun relative to LSR
    v_correct = vlsr[0]*cos(b)*cos(l) + \
        vlsr[1]*cos(b)*sin(l) + \
        vlsr[2]*sin(b)
    vgsr = lsr + v_correct

    return vgsr
