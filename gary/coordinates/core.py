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

def gal_xyz_to_hel(xyz, vxyz=None,
                   vcirc=default_vcirc, vlsr=default_vlsr, xsun=default_xsun):
    r""" Convert Galactocentric cartesian coordinates to Heliocentric
        spherical coordinates. Uses a right-handed cartesian system,
        with the Sun at :math:`x \approx -8~{\rm kpc}`.

        Parameters
        ----------
        xyz : :class:`~astropy.units.Quantity`, iterable
            Cartesian x,y,z coordinates. This should either be a single
            :class:`~astropy.units.Quantity` object with shape (3,N), or an iterable
            object with 3 :class:`~astropy.units.Quantity` objects as elements.
        vxyz : :class:`~astropy.units.Quantity` (optional)
            Cartesian velocity components (U,V,W). This should either be a single
            :class:`~astropy.units.Quantity` object with shape (3,N), or an iterable
            object with 3 :class:`~astropy.units.Quantity` objects as elements.
        vcirc : :class:`~astropy.units.Quantity`
            Circular velocity of the Sun.
        vlsr : :class:`~astropy.units.Quantity`
            Velocity of the Sun relative to the local standard
            of rest (LSR).
        xsun : :class:`~astropy.units.Quantity`
            Position of the Sun on the Galactic x-axis.

        Returns
        -------
        lbd : :class:`~astropy.coordinates.SkyCoord`
            The position in heliocentric coordinates as an Astropy coordinates
            :class:`~astropy.coordinates.SkyCoord` object.
        pmv : tuple (optional)
            If velocity information is provided, also return a tuple containing
            the proper motions (in Galactic coordinates) and radial velocity,
            all as :class:`~astropy.units.Quantity` objects.

    """

    # unpack positions
    try:
        x,y,z = xyz
    except ValueError:
        if len(xyz.shape) > 1 and xyz.shape[0] > xyz.shape[1]:
            raise ValueError("Could not unpack positions -- the shape looks"
                             " transposed. Should have shape (3,N).")
        else:
            raise ValueError("Failed to unpack positions with shape {}."
                             " Should have shape (3,N).".format(xyz.shape))

    # transform to heliocentric cartesian
    x = x - xsun

    # transform from cartesian to spherical
    d = np.sqrt(x**2 + y**2 + z**2)
    l = coord.Angle(np.arctan2(y, x)).wrap_at(360*u.deg).to(u.degree)
    b = coord.Angle(90*u.degree - np.arccos(z/d)).to(u.degree)
    lbd = coord.SkyCoord(l=l, b=b, distance=d)

    if vxyz is not None:
        if len(vxyz) != len(xyz):
            raise ValueError("Shape of velocity should match position.")

        # unpack velocities
        vx,vy,vz = vxyz

        # transform to heliocentric cartesian
        vy = vy - vcirc

        # correct for motion of Sun relative to LSR
        vx = vx - vlsr[0]
        vy = vy - vlsr[1]
        vz = vz - vlsr[2]

        # transform cartesian velocity to spherical
        d_xy = np.sqrt(x**2 + y**2)
        vr = (vx*x + vy*y + vz*z) / d  # velocity
        omega_l = -(vx*y - x*vy) / d_xy**2  # angular velocity
        omega_b = -(z*(x*vx + y*vy) - d_xy**2*vz) / (d**2 * d_xy)  # angular velocity

        mul = (omega_l.decompose()*u.rad).to(u.milliarcsecond / u.yr)
        mub = (omega_b.decompose()*u.rad).to(u.milliarcsecond / u.yr)

        return lbd, (mul,mub,vr)

    return lbd

def hel_to_gal_xyz(coordinate, pm=None, vr=None,
                   vcirc=default_vcirc, vlsr=default_vlsr, xsun=default_xsun):
    r""" Convert Heliocentric spherical coordinates to Galactocentric
        cartesian coordinates. Uses a right-handed cartesian system,
        typically with the Sun at :math:`x \approx -8~{\rm kpc}`.

        Parameters
        ----------
        coordinate : :class:`~astropy.coordinates.SkyCoord`
            An Astropy SkyCoord object or anything object that can be passed
            to the SkyCoord initializer. Must have a distance defined.
        pm : :class:`~astropy.units.Quantity` (optional)
            Proper motion in l, b. Should have shape (2,N).
        vr : :class:`~astropy.units.Quantity` (optional)
            Barycentric radial velocity. Should have shape (1,N) or (N,).
        vcirc : :class:`~astropy.units.Quantity`
            Circular velocity of the Sun.
        vlsr : :class:`~astropy.units.Quantity`
            Velocity of the Sun relative to the local standard
            of rest (LSR).
        xsun : :class:`~astropy.units.Quantity`
            Position of the Sun on the Galactic x-axis.

        Returns
        -------
        xyz : :class:`~astropy.units.Quantity`, iterable
            Cartesian x,y,z coordinates. A :class:`~astropy.units.Quantity`
            object with shape (3,N).
        vxyz : :class:`~astropy.units.Quantity` (optional)
            Cartesian velocity components (U,V,W). A :class:`~astropy.units.Quantity`
            object with shape (3,N).
    """

    c = coord.SkyCoord(coordinate)
    g = c.galactic
    l,b,d = g.l, g.b, g.distance

    # spherical to cartesian
    x = d*np.cos(b)*np.cos(l)
    y = d*np.cos(b)*np.sin(l)
    z = d*np.sin(b)

    if pm is not None:
        if vr is None:
            raise ValueError("If proper motions are specified, radial velocity must"
                             " also be specified.")

        # unpack velocities
        mul,mub = pm
        vr = np.squeeze(vr)

        omega_l = -mul.to(u.rad/u.s).value/u.s
        omega_b = -mub.to(u.rad/u.s).value/u.s

        vx = x/d*vr + y*omega_l + z*np.cos(l)*omega_b
        vy = y/d*vr - x*omega_l + z*np.sin(l)*omega_b
        vz = z/d*vr - d*np.cos(b)*omega_b

        # transform to galactocentric cartesian
        vy = vy + vcirc

        # correct for motion of Sun relative to LSR
        vx = vx + vlsr[0]
        vy = vy + vlsr[1]
        vz = vz + vlsr[2]

        # transform to galactocentric cartesian
        x = x + xsun

        # Workaround because Quantities can't be vstack'd
        return (np.vstack((x.value,y.value,z.value))*x.unit,
                np.vstack((vx.value,vy.value,vz.value))*vx.unit)

    else:
        # transform to galactocentric cartesian
        x = x + xsun
        return np.vstack((x.value,y.value,z.value))*x.unit
