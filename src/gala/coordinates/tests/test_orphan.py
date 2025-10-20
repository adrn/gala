"""
Test the coordinates class that represents the plane of orbit of the Sgr dwarf galaxy.
"""

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from ..orphan import OrphanKoposov19, OrphanNewberg10


def test_table():
    """Test the transformation code against table 2 values from
    Newberg et al. 2010 (below)
    """

    names = ["l", "b", "db", "Lambda", "Beta", "g0", "dg0"]
    table = """255 48.5 0.7 22.34 0.08 17.1 0.1
245 52.0 0.7 15.08 0.56 0. 0.
235 53.5 0.7 8.86 0.21 0. 0.
225 54.0 0.7 2.95 -0.23 17.6 0.2
215 54.0 0.7 -2.93 -0.33 17.9 0.1
205 53.5 0.7 -8.85 -0.09 18.0 0.1
195 52.0 0.7 -15.08 0.05 0. 0.
185 50.5 0.7 -21.42 1.12 18.6 0.1
175 47.5 0.7 -28.59 1.88 0. 0.
171 45.8 1.0 -31.81 2.10 0. 0."""

    table = ascii.read(table, names=names)

    for line in table:
        galactic = coord.Galactic(l=line["l"] * u.deg, b=line["b"] * u.deg)

        orp = galactic.transform_to(OrphanNewberg10())
        true_orp = OrphanNewberg10(
            phi1=line["Lambda"] * u.deg, phi2=line["Beta"] * u.deg
        )

        # TODO: why does this suck so badly?
        assert true_orp.separation(orp) < 20 * u.arcsec


def test_kopsov():
    tbl = Table.read(get_pkg_data_filename("sergey_orphan.txt"), format="ascii")
    c = coord.SkyCoord(ra=tbl["ra"] * u.deg, dec=tbl["dec"] * u.deg)
    orp_gc = c.transform_to(OrphanKoposov19())
    assert np.percentile(orp_gc.phi2.degree, 95) < 5
