import astropy.coordinates as coord

__all__ = ['reflex_correct']


def reflex_correct(coords, galactocentric_frame=None):
    """Correct the input Astropy coordinate object for solar reflex motion.

    The input coordinate instance must have distance and radial velocity information.
    So, if the radial velocity is not known, fill the radial velocity values with zeros
    to reflex-correct the proper motions.

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
        The Astropy coordinate object with position and velocity information.
    galactocentric_frame : `~astropy.coordinates.Galactocentric` (optional)
        To change properties of the Galactocentric frame, like the height of the
        sun above the midplane, or the velocity of the sun in a Galactocentric
        intertial frame, set arguments of the
        `~astropy.coordinates.Galactocentric` object and pass in to this
        function with your coordinates.

    Returns
    -------
    coords : `~astropy.coordinates.SkyCoord`
        The coordinates in the same frame as input, but with solar motion
        removed.

    """
    c = coord.SkyCoord(coords)

    # If not specified, use the Astropy default Galactocentric frame
    if galactocentric_frame is None:
        galactocentric_frame = coord.Galactocentric()

    v_sun = galactocentric_frame.galcen_v_sun

    observed = c.transform_to(galactocentric_frame)
    rep = observed.cartesian.without_differentials()
    rep = rep.with_differentials(observed.cartesian.differentials['s'] + v_sun)
    fr = galactocentric_frame.realize_frame(rep).transform_to(c.frame)
    return coord.SkyCoord(fr)
