# coding: utf-8

""" Transform a proper motion to/from Galactic to/from ICRS coordinates """

from __future__ import division, print_function

# Standard library
import warnings

# Third-party
import numpy as np
import astropy.coordinates as coord

__all__ = ['transform_proper_motion', 'pm_gal_to_icrs', 'pm_icrs_to_gal']

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
    url = "http://docs.astropy.org/en/stable/coordinates/velocities.html"
    warnings.warn("This function is now deprecated. Use the velocity "
                  "transformation functionality in Astropy instead. For more "
                  "information, see: {0}".format(url), DeprecationWarning)

    if hasattr(coordinate, 'frame'):
        coordinate = coordinate.frame

    frame_cls = coordinate.__class__
    c = frame_cls(coordinate.data.with_differentials(
        coord.UnitSphericalCosLatDifferential(*pm)))
    new_c = c.transform_to(new_frame)
    diff = new_c.data.differentials['s']
    pm = np.vstack((diff.d_lon_coslat.value, diff.d_lat.value)) * diff.d_lat.unit

    return pm.reshape((2,) + c.shape)

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

    # Note: Deprecation warning gets emitted from calling
    #       transform_proper_motion below.
    g = coordinate.transform_to(coord.Galactic)
    return transform_proper_motion(g, pm, coord.ICRS)

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

    # Note: Deprecation warning gets emitted from calling
    #       transform_proper_motion below.
    i = coordinate.transform_to(coord.ICRS)
    return transform_proper_motion(i, pm, coord.Galactic)

