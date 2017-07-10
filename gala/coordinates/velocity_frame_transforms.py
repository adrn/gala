# coding: utf-8

""" Miscellaneous astronomical velocity transformations. """

from __future__ import division, print_function


# Standard library
import warnings

import astropy.coordinates as coord
import astropy.units as u

__all__ = ["vgal_to_hel", "vhel_to_gal", "vgsr_to_vhel", "vhel_to_vgsr"]

def _get_vproj(c, vsun):
    gal = c.transform_to(coord.Galactic)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()
    return coord.CartesianRepresentation(vsun).dot(unit_vector)

def vgsr_to_vhel(coordinate, vgsr, vsun=None):
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
    vsun : :class:`~astropy.units.Quantity`
        Full-space velocity of the sun in a Galactocentric frame. By default,
        uses the value assumed by Astropy in
        `~astropy.coordinates.Galactocentric`.

    Returns
    -------
    vhel : :class:`~astropy.units.Quantity`
        Radial velocity in a barycentric rest frame.

    """

    if vsun is None:
        vsun = coord.Galactocentric.galcen_v_sun.to_cartesian().xyz

    return vgsr - _get_vproj(coordinate, vsun)

def vhel_to_vgsr(coordinate, vhel, vsun):
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
    vsun : :class:`~astropy.units.Quantity`
        Full-space velocity of the sun in a Galactocentric frame. By default,
        uses the value assumed by Astropy in
        `~astropy.coordinates.Galactocentric`.

    Returns
    -------
    vgsr : :class:`~astropy.units.Quantity`
        Radial velocity in a galactocentric rest frame.

    """

    if vsun is None:
        vsun = coord.Galactocentric.galcen_v_sun.to_cartesian().xyz

    return vhel + _get_vproj(coordinate, vsun)

# ----------------------------------------------------------------------------

def vgal_to_hel(coordinate, vxyz, vcirc=None, vlsr=None,
                galactocentric_frame=None):
    r"""THIS FUNCTION IS DEPRECATED!

    Use the `velocity transformations
    <http://docs.astropy.org/en/stable/coordinates/velocities.html>`_ from
    Astropy instead.

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
    velocity : :class:`~astropy.units.Quantity`, iterable
        Cartesian velocity components (vx,vy,vz). This should either
        be a single Quantity object with shape (3,N), or an iterable object with
        3 Quantity objects as elements.
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
    """

    url = "http://docs.astropy.org/en/stable/coordinates/velocities.html"
    warnings.warn("This function is now deprecated. Use the velocity "
                  "transformation functionality in Astropy instead. For more "
                  "information, see: {0}".format(url), DeprecationWarning)

    if hasattr(coordinate, 'frame'):
        frame = coordinate.frame
    else:
        frame = coordinate

    if vcirc is None:
        vcirc = 220 * u.km/u.s # default in Astropy

    if vlsr is None:
        vlsr = [11.1, 12.24, 7.25] * u.km/u.s # default in Astropy

    v_sun = vlsr + [0,1,0]*vcirc

    if galactocentric_frame is None:
        galactocentric_frame = coord.Galactocentric

    kwargs = dict([(k, getattr(galactocentric_frame, k))
                   for k in galactocentric_frame.frame_attributes])
    kwargs['galcen_v_sun'] = coord.CartesianDifferential(*v_sun)

    gc_no_data = coord.Galactocentric(**kwargs)
    gc = coordinate.transform_to(gc_no_data)
    new_rep = gc.data.with_differentials(coord.CartesianDifferential(*vxyz))

    gc = coord.Galactocentric(new_rep, **kwargs)
    new_c = gc.transform_to(frame)

    diff = new_c.represent_as('spherical', 'sphericalcoslat').differentials['s']
    return (diff.d_lon_coslat, diff.d_lat, diff.d_distance)

def vhel_to_gal(coordinate, pm, rv, vcirc=None, vlsr=None,
                galactocentric_frame=None):
    r"""THIS FUNCTION IS DEPRECATED!

    Use the `velocity transformations
    <http://docs.astropy.org/en/stable/coordinates/velocities.html>`_ from
    Astropy instead.

    The frame of the input coordinate determines how to interpret the given
    proper motions. For example, if the input coordinate is in the ICRS frame,
    the input velocity is assumed to be as well.

    Parameters
    ----------
    coordinate : :class:`~astropy.coordinates.SkyCoord`, :class:`~astropy.coordinates.BaseCoordinateFrame`
        This is most commonly a :class:`~astropy.coordinates.SkyCoord` object,
        but alternatively, it can be any coordinate frame object that is
        transformable to the Galactocentric frame.
    pm : :class:`~astropy.units.Quantity` or iterable of :class:`~astropy.units.Quantity` objects
        Proper motion in the same frame as the coordinate. For example, if your
        input coordinate is in :class:`~astropy.coordinates.ICRS`, then the
        proper motion is assumed to be in this frame as well. The order of
        elements should always be proper motion in (longitude, latitude), and
        should have shape (2,N). The longitude component is assumed to have the
        cosine of the latitude already multiplied in, so that in ICRS, for
        example, this would be :math:`\mu_\alpha\cos\delta`.
    rv : :class:`~astropy.units.Quantity`
        Barycentric radial velocity. Should have shape (1,N) or (N,).
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

    """

    url = "http://docs.astropy.org/en/stable/coordinates/velocities.html"
    warnings.warn("This function is now deprecated. Use the velocity "
                  "transformation functionality in Astropy instead. For more "
                  "information, see: {0}".format(url), DeprecationWarning)

    if hasattr(coordinate, 'frame'):
        frame = coordinate.frame
    else:
        frame = coordinate

    frame_cls = frame.__class__
    c = frame_cls(coordinate.data.with_differentials(
        coord.SphericalCosLatDifferential(pm[0], pm[1], rv)))

    if vcirc is None:
        vcirc = 220 * u.km/u.s # default in Astropy

    if vlsr is None:
        vlsr = [11.1, 12.24, 7.25] * u.km/u.s # default in Astropy

    v_sun = vlsr + [0,1,0]*vcirc

    if galactocentric_frame is None:
        galactocentric_frame = coord.Galactocentric

    kwargs = dict([(k, getattr(galactocentric_frame, k))
                   for k in galactocentric_frame.frame_attributes])
    kwargs['galcen_v_sun'] = coord.CartesianDifferential(*v_sun)
    gc_no_data = coord.Galactocentric(**kwargs)

    new_c = c.transform_to(gc_no_data)
    return new_c.data.differentials['s'].d_xyz
