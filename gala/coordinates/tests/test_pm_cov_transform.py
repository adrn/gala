# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np

# This project
from ..orphan import KoposovOrphan
from ..pm_cov_transform import transform_pm_cov


def test_transform():
    ra, dec, pmra, pmdec = np.load(get_pkg_data_filename('c_pm.npy'))
    cov = np.load(get_pkg_data_filename('pm_cov.npy'))

    print(len(ra), cov.shape)
