# coding: utf-8

""" """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from numpy import cos, sin

import astropy.coordinates as coord
import astropy.units as u

__all__ = ["vgsr_to_vhel", "vhel_to_vgsr", "gal_xyz_to_hel", "hel_to_gal_xyz"]

# This is the default circular velocity and LSR peculiar velocity of the Sun
# TODO: make this a config item
default_vcirc = 220.*u.km/u.s
default_vlsr = [10., 5.25, 7.17]*u.km/u.s
default_xsun = -8.*u.kpc

def vgsr_to_vhel(coordinate, vgsr, vcirc=default_vcirc, vlsr=default_vlsr):
    """ Convert a radial velocity in the Galactic standard of rest (GSR) to
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

def vhel_to_vgsr(coordinate, vhel, vcirc=default_vcirc, vlsr=default_vlsr):
    """ Convert a velocity from a heliocentric radial velocity to
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

def vgal_to_hel(coordinate, vxyz, vcirc=default_vcirc, vlsr=default_vlsr):
    r"""
    Convert a Galactocentric, cartesian velocity to a Heliocentric velocity in
    spherical coordinates (e.g., proper motion and radial velocity).

    Parameters
    ----------
    coordinate : `~astropy.coordinates.SkyCoord`, `~astropy.coordinates.BaseCoordinateFrame`
        This is most commonly a `~astropy.coordinates.Galactocentric` Astropy
        coordinate, but alternatively, it can be any coordinate object that is
        transformable to the Galactocentric frame.
    vxyz : :class:`~astropy.units.Quantity`, iterable
        Cartesian velocity components (U,V,W). This should either be a single
        :class:`~astropy.units.Quantity` object with shape (3,N), or an iterable
        object with 3 :class:`~astropy.units.Quantity` objects as elements.
    vcirc : :class:`~astropy.units.Quantity` (optional)
        Circular velocity of the Sun.
    vlsr : :class:`~astropy.units.Quantity` (optional)
        Velocity of the Sun relative to the local standard
        of rest (LSR).

    Returns
    -------
    pmv : tuple
        A tuple containing the proper motions (in Galactic coordinates) and
        radial velocity, all as :class:`~astropy.units.Quantity` objects.

    """

    # make sure we have a Galactocentric coordinate
    c = coord.SkyCoord(coordinate)
    gc = c.transform_to(coord.Galactocentric)
    x,y,z = gc.cartesian.xyz

    if vxyz.shape != gc.cartesian.xyz.shape:
        raise ValueError("Shape of velocity must match position.")

    # unpack velocities
    vx,vy,vz = vxyz

    # transform to heliocentric cartesian
    vy = vy - vcirc

    # correct for motion of Sun relative to LSR
    vx = vx - vlsr[0]
    vy = vy - vlsr[1]
    vz = vz - vlsr[2]

    # transform cartesian velocity to spherical
    d = np.sqrt(x**2 + y**2 + z**2)
    d_xy = np.sqrt(x**2 + y**2)
    vr = (vx*x + vy*y + vz*z) / d  # velocity
    omega_l = -(vx*y - x*vy) / d_xy**2  # angular velocity
    omega_b = -(z*(x*vx + y*vy) - d_xy**2*vz) / (d**2 * d_xy)  # angular velocity

    mul = (omega_l.decompose()*u.rad).to(u.milliarcsecond / u.yr)
    mub = (omega_b.decompose()*u.rad).to(u.milliarcsecond / u.yr)

    return mul,mub,vr

def vhel_to_gal(coordinate, pm, rv, vcirc=default_vcirc, vlsr=default_vlsr):
    r"""
    Convert a Heliocentric, spherical velocity to a Galactocentric,
    cartesian velocity.

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`
        This is most commonly a `~astropy.coordinates.Galactocentric` Astropy
        coordinate, but alternatively, it can be any coordinate object that is
        transformable to the Galactocentric frame.
    pm : iterable of :class:`~astropy.units.Quantity`s
        Proper motion in l, b. Should have shape (2,N).
    vr : :class:`~astropy.units.Quantity` (optional)
        Barycentric radial velocity. Should have shape (1,N) or (N,).
    vcirc : :class:`~astropy.units.Quantity`
        Circular velocity of the Sun.
    vlsr : :class:`~astropy.units.Quantity`
        Velocity of the Sun relative to the local standard
        of rest (LSR).

    Returns
    -------
    vxyz : :class:`~astropy.units.Quantity` (optional)
        Cartesian velocity components (U,V,W). A :class:`~astropy.units.Quantity`
        object with shape (3,N).
    """

    c = coord.SkyCoord(coordinate)
    l,b,d = c.galactic.l, c.galactic.b, c.galactic.distance
    gc = c.transform_to(coord.Galactocentric)
    x,y,z = gc.cartesian.xyz

    if pm.shape[1] != rv.size or pm.shape[1] != coordinate.shape[1]:
        raise ValueError("Length of proper motion and radial velocity must"
                         " be consistent with axis=1 size of coordinate.")

    if rv is None:
        raise ValueError("If proper motions are specified, radial velocity must"
                         " also be specified.")

    # unpack velocities
    mul,mub = pm
    rv = np.squeeze(rv)

    omega_l = -mul.to(u.rad/u.s).value/u.s
    omega_b = -mub.to(u.rad/u.s).value/u.s

    vx = x/d*rv + y*omega_l + z*np.cos(l)*omega_b
    vy = y/d*rv - x*omega_l + z*np.sin(l)*omega_b
    vz = z/d*rv - d*np.cos(b)*omega_b

    # transform to galactocentric cartesian
    vy = vy + vcirc

    # correct for motion of Sun relative to LSR
    vx = vx + vlsr[0]
    vy = vy + vlsr[1]
    vz = vz + vlsr[2]

    # Workaround because Quantities can't be vstack'd
    return np.vstack((vx.value,vy.value,vz.value))*vx.unit
