# coding: utf-8

""" Transform a proper motion to/from Galactic to/from ICRS coordinates """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import astropy.coordinates as coord

__all__ = ['pm_gal_to_icrs', 'pm_icrs_to_gal']

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
