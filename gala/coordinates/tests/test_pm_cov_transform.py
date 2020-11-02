# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import pytest

# This project
from ..orphan import OrphanKoposov19
from ..pm_cov_transform import transform_pm_cov

sky_offset_frame = coord.SkyOffsetFrame(
    origin=coord.ICRS(ra="20d", dec="30d"),
    rotation=135.7 * u.deg
)


def setup_function(fn):
    ra, dec, pmra, pmdec = np.load(get_pkg_data_filename('c_pm.npy'))
    c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                       pm_ra_cosdec=pmra*u.mas/u.yr,
                       pm_dec=pmdec*u.mas/u.yr)
    cov = np.load(get_pkg_data_filename('pm_cov.npy'))

    fn.c = c
    fn.cov = cov


@pytest.mark.parametrize("to_frame", [coord.Galactic, coord.Galactic(),
                                      coord.Supergalactic, coord.Supergalactic(),
                                      OrphanKoposov19, OrphanKoposov19(),
                                      sky_offset_frame])
def test_transform(to_frame):
    c = test_transform.c
    cov = test_transform.cov

    # First, don't validate, just check input paths:
    transform_pm_cov(c[0], cov[0], to_frame)
    transform_pm_cov(c[:4], cov[:4], to_frame)

    with pytest.raises(ValueError):
        transform_pm_cov(c[:4], cov[:8], to_frame)

    with pytest.raises(ValueError):
        transform_pm_cov(c[0], cov[0, :1], to_frame)

    new_cov1 = transform_pm_cov(c[0], cov[0], to_frame)
    new_cov2 = np.squeeze(transform_pm_cov(c[0:1], cov[0:1], to_frame))
    assert np.allclose(new_cov1, new_cov2)


@pytest.mark.parametrize("to_frame", [coord.Galactic, coord.Galactic(),
                                      coord.Supergalactic, coord.Supergalactic(),
                                      OrphanKoposov19, OrphanKoposov19(),
                                      sky_offset_frame])
def test_transform_correctness(to_frame):
    c = test_transform_correctness.c[:4]
    cov = test_transform_correctness.cov[:4]

    # generate proper motion samples and transform the samples:
    pm = np.vstack((c.pm_ra_cosdec.value,
                    c.pm_dec.value)).T
    rnd = np.random.RandomState(42)

    for i in range(len(c)):
        pm_samples = rnd.multivariate_normal(pm[i], cov[i],
                                             size=2**16)
        c1 = coord.SkyCoord(ra=[c[i].ra.value]*pm_samples.shape[0] * u.deg,
                            dec=[c[i].dec.value]*pm_samples.shape[0] * u.deg,
                            pm_ra_cosdec=pm_samples[:, 0]*u.mas/u.yr,
                            pm_dec=pm_samples[:, 1]*u.mas/u.yr)
        new_c1 = c1.transform_to(to_frame)

        dsph = new_c1.represent_as(coord.SphericalRepresentation,
                                   coord.SphericalCosLatDifferential).differentials['s']
        new_pm_samples = np.vstack((dsph.d_lon_coslat.value,
                                    dsph.d_lat.value))
        cov_est = np.cov(new_pm_samples)
        cov_trans = transform_pm_cov(c[i], cov[i], to_frame)
        assert np.allclose(cov_est, cov_trans, atol=1e-2)
        assert np.allclose(np.sort(np.linalg.eigvals(cov[i])),
                           np.sort(np.linalg.eigvals(cov_trans)))
