# coding: utf-8
"""
    Test conversions in propermotion.py
"""

from __future__ import absolute_import, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import tempfile

import numpy as np

import astropy.coordinates as coord
import astropy.units as u

from ..propermotion import *

_txt = """ # from here: http://www.astrostudio.org/xhipreadme.html
# HIPID ra dec pmra pmdec pml pmb
1 0.00091185 1.08901332    -4.58    -1.61    -4.85     0.29
2 0.00379738 -19.49883738   179.70     1.40   104.99  -145.85
3 0.00500794 38.85928608     4.28    -3.42     3.44    -4.26
10241 32.92989805 -31.39849677    40.91    18.19   -24.11    37.72
10242 32.93085553 3.45271681    18.94    -1.75    17.12     8.29
10243 32.93296084 60.71236365     0.85    -1.23     1.18    -0.91
19265 61.93883644 80.97646929    -7.44    -2.57    -3.31    -7.14
19266 61.93960267 4.31970083    -7.87    -3.75    -1.68    -8.55
19267 61.94490325 2.72570815    11.89   -33.83    34.41   -10.08
"""

class TestPMConvert(object):

    def setup(self):
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(_txt)
            temp.flush()
            temp.seek(0)
            self.data = np.genfromtxt(temp, names=True, skiprows=1)

    def test_pm_gal_to_icrs(self):

        # test a single entry
        row = self.data[0]
        c = coord.SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg)
        muad = [row['pmra'],row['pmdec']]*u.mas/u.yr
        mulb = [row['pml'],row['pmb']]*u.mas/u.yr

        trans_muad = pm_gal_to_icrs(c, mulb)[:,0]
        assert np.allclose(muad, trans_muad, atol=1E-2)

        # multiple entries
        c = coord.SkyCoord(ra=self.data['ra']*u.deg, dec=self.data['dec']*u.deg)
        muad = np.vstack((self.data['pmra'],self.data['pmdec']))*u.mas/u.yr
        mulb = np.vstack((self.data['pml'],self.data['pmb']))*u.mas/u.yr

        trans_muad = pm_gal_to_icrs(c, mulb)
        assert np.allclose(muad, trans_muad, atol=1E-2)

    def test_pm_icrs_to_gal(self):

        # test a single entry
        row = self.data[0]
        c = coord.SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg)
        muad = [row['pmra'],row['pmdec']]*u.mas/u.yr
        mulb = [row['pml'],row['pmb']]*u.mas/u.yr

        trans_mulb = pm_icrs_to_gal(c, muad)[:,0]
        assert np.allclose(mulb, trans_mulb, atol=1E-2)

        # multiple entries
        c = coord.SkyCoord(ra=self.data['ra']*u.deg, dec=self.data['dec']*u.deg)
        muad = np.vstack((self.data['pmra'],self.data['pmdec']))*u.mas/u.yr
        mulb = np.vstack((self.data['pml'],self.data['pmb']))*u.mas/u.yr

        trans_mulb = pm_icrs_to_gal(c, muad)
        assert np.allclose(mulb, trans_mulb, atol=1E-2)

