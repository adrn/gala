# Third-party
import astropy.coordinates as coord
import astropy.units as u

# This package
from ..reflex import reflex_correct


def test_reflex():
    c = coord.SkyCoord(ra=162*u.deg,
                       dec=-17*u.deg,
                       distance=172*u.pc,
                       pm_ra_cosdec=-11*u.mas/u.yr,
                       pm_dec=4*u.mas/u.yr,
                       radial_velocity=110*u.km/u.s)

    # First, test execution but don't validate
    reflex_correct(c)
    with coord.galactocentric_frame_defaults.set('v4.0'):
        reflex_correct(c, coord.Galactocentric(z_sun=0*u.pc))

    # Reflext correct the observed, Reid & Brunthaler (2004) Sgr A* measurements
    # and make sure the corrected velocity is close to zero
    # https://ui.adsabs.harvard.edu/abs/2004ApJ...616..872R/abstract
    # also using
    # https://ui.adsabs.harvard.edu/abs/2018RNAAS...2d.210D/abstract
    # https://ui.adsabs.harvard.edu/abs/2018A%26A...615L..15G/abstract
    vsun = coord.CartesianDifferential([12.9, 245.6, 7.78] * u.km/u.s)
    with coord.galactocentric_frame_defaults.set('v4.0'):
        galcen_fr = coord.Galactocentric(galcen_distance=8.122*u.kpc,
                                         galcen_v_sun=vsun,
                                         z_sun=20.8*u.pc)

    sgr_Astar_obs = coord.SkyCoord(ra=galcen_fr.galcen_coord.ra,
                                   dec=galcen_fr.galcen_coord.dec,
                                   distance=galcen_fr.galcen_distance,
                                   pm_ra_cosdec=-3.151*u.mas/u.yr,
                                   pm_dec=-5.547*u.mas/u.yr,
                                   radial_velocity=-12.9*u.km/u.s)

    new_c = reflex_correct(sgr_Astar_obs, galcen_fr)
    assert u.allclose(new_c.pm_ra_cosdec, 0*u.mas/u.yr, atol=1e-2*u.mas/u.yr)
    assert u.allclose(new_c.pm_dec, 0*u.mas/u.yr, atol=1e-2*u.mas/u.yr)
    assert u.allclose(new_c.radial_velocity, 0*u.km/u.s, atol=1e-1*u.km/u.s)
