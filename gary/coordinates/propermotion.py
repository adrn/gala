# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import astropy.coordinates as coord

__all__ = ['pm_gal_to_icrs', 'pm_icrs_to_gal']

def pm_gal_to_icrs(coordinate, mu, cosb=False):
    """ Convert proper motion in Galactic coordinates (l,b) to
        ICRS coordinates (RA, Dec).

        Parameters
        ----------
        coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
            An instance of an Astropy coordinate object. Can be in any
            frame that is transformable to ICRS coordinates.
        mu : :class:`~astropy.units.Quantity`, iterable
            Full description of proper motion in Galactic longitude and
            latitude. Can either be a tuple of two
            :class:`~astropy.units.Quantity` objects or a single
            :class:`~astropy.units.Quantity` with shape (2,N).
            If proper motion of longitude is pre-multipled by cosine of
            Galactic latitude, use the keyword argument `cosb=True` to
            automatically handle this.
        cosb : bool (optional)
            Specify whether the input proper motion in longitude has been
            pre-multiplied by cosine of the latitude. Default is False.

        Returns
        -------
        pm : :class:`~astropy.units.Quantity`
            An astropy :class:`~astropy.units.Quantity` object specifying the
            proper motion vector array in ICRS coordinates. Will have shape
            (2,N).
    """

    g = coordinate.transform_to(coord.Galactic)
    i = coordinate.transform_to(coord.ICRS)

    mul,mub = map(np.atleast_1d, mu)
    if cosb is False:
        mulcosb = mul*np.cos(g.b)
    else:
        mulcosb = mul

    n = len(mulcosb)

    # coordinates of NGP
    ag = coord.Galactic._ngp_J2000.ra
    dg = coord.Galactic._ngp_J2000.dec

    # used for the transformation matrix from Bovy (2011)
    cosphi = (np.sin(dg) - np.sin(i.dec)*np.sin(g.b)) / np.cos(g.b) / np.cos(i.dec)
    sinphi = np.sin(i.ra - ag) * np.cos(dg) / np.cos(g.b)

    R = np.zeros((2,2,n))
    R[0,0] = cosphi
    R[0,1] = -sinphi
    R[1,0] = sinphi
    R[1,1] = cosphi

    mu = np.vstack((mulcosb.value, mub.to(mulcosb.unit).value))
    new_mu = np.zeros_like(mu)
    for i in range(n):
        new_mu[:,i] = R[...,i].dot(mu[:,i])

    return new_mu*mulcosb.unit

def pm_icrs_to_gal(coordinate, mu, cosdec):
    """ Convert proper motion in ICRS coordinates (RA, Dec) to
        Galactic coordinates (l,b).

        Parameters
        ----------
        coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
            An instance of an Astropy coordinate object. Can be in any
            frame that is transformable to ICRS coordinates.
        mu : :class:`~astropy.units.Quantity`, iterable
            Full description of proper motion in Right ascension (RA) and
            declination (Dec). Can either be a tuple of two
            :class:`~astropy.units.Quantity` objects or a single
            :class:`~astropy.units.Quantity` with shape (2,N).
            If proper motion of RA is pre-multipled by cosine of
            Dec, use the keyword argument `cosdec=True` to
            automatically handle this.
        cosdec : bool (optional)
            Specify whether the input proper motion in RA has been
            pre-multiplied by cosine of the Dec. Default is False.

        Returns
        -------
        pm : :class:`~astropy.units.Quantity`
            An astropy :class:`~astropy.units.Quantity` object specifying the
            proper motion vector array in Galactic coordinates. Will have shape
            (2,N).
    """

    g = coordinate.transform_to(coord.Galactic)
    i = coordinate.transform_to(coord.ICRS)

    mua,mud = map(np.atleast_1d, mu)
    if cosdec is False:
        muacosd = mua*np.cos(i.dec)
    else:
        muacosd = mua

    n = len(muacosd)

    # coordinates of NGP
    ag = coord.Galactic._ngp_J2000.ra
    dg = coord.Galactic._ngp_J2000.dec

    # used for the transformation matrix from Bovy (2011)
    cosphi = (np.sin(dg) - np.sin(i.dec)*np.sin(g.b)) / np.cos(g.b) / np.cos(i.dec)
    sinphi = np.sin(i.ra - ag) * np.cos(dg) / np.cos(g.b)

    R = np.zeros((2,2,n))
    R[0,0] = cosphi
    R[0,1] = sinphi
    R[1,0] = -sinphi
    R[1,1] = cosphi

    mu = np.vstack((muacosd.value, mud.to(muacosd.unit).value))
    new_mu = np.zeros_like(mu)
    for i in range(n):
        new_mu[:,i] = R[...,i].dot(mu[:,i])

    return new_mu*muacosd.unit
