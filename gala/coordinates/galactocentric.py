# Third-party
import astropy.coordinates as coord

__all__ = ['get_galactocentric2019']


def get_galactocentric2019():
    """
    DEPRECATED!

    Use astropy v4.0 or later:

        >>> with coord.galactocentric_frame_defaults.set('v4.0'):
        ...     galcen = coord.Galactocentric()

    """

    with coord.galactocentric_frame_defaults.set('v4.0'):
        galcen = coord.Galactocentric()

    return galcen
