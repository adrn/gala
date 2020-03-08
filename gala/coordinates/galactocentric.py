# Third-party
import astropy.coordinates as coord
import astropy.units as u

__all__ = ['get_galactocentric2019']


def get_galactocentric2019():
    """
    References
    ----------
    Galactic center sky position:
    - `Reid & Brunthaler (2004) <https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R/abstract>`_

    Galactic center distance:
    - `GRAVITY collaboration (2018) <https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G/abstract>`_

    Solar velocity:
    - `Drimmel & Poggio (2018) <https://ui.adsabs.harvard.edu/abs/2018RNAAS...2d.210D/abstract>`_
    - `Reid & Brunthaler (2004) <https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R/abstract>`_
    - `GRAVITY collaboration (2018) <https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G/abstract>`_

    Solar height above midplane:
    - `Bennett & Bovy (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.482.1417B/abstract>`_

    Returns
    -------
    galcen_frame : `~astropy.coordinates.Galactocentric`
        The modern (2019) Galactocentric reference frame.

    """

    with coord.galactocentric_frame_defaults.set('v4.0'):
        galcen = coord.Galactocentric()

    return galcen
