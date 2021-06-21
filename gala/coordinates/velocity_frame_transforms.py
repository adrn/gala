""" Miscellaneous astronomical velocity transformations. """

import astropy.coordinates as coord

__all__ = ["vgsr_to_vhel", "vhel_to_vgsr"]


def _get_vproj(c, vsun):
    gal = c.transform_to(coord.Galactic())
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
        galcen = coord.Galactocentric()
        vsun = galcen.galcen_v_sun.to_cartesian().xyz

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
        galcen = coord.Galactocentric()
        vsun = galcen.galcen_v_sun.to_cartesian().xyz

    return vhel + _get_vproj(coordinate, vsun)
