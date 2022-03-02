# coding: utf-8

# Standard library
import os
from math import factorial as _factorial

# Third-party
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import pytest

# Project
from gala._cconfig import GSL_ENABLED
from ..core import compute_coeffs_discrete

if not GSL_ENABLED:
    pytest.skip("skipping SCF tests: they depend on GSL",
                allow_module_level=True)

# Compare coefficients computed with Fortran to Biff


def factorial(x):
    return _factorial(int(x))


@pytest.mark.parametrize("basename", [
    'hernquist'
])
def test_coeff(basename):
    nmax = 6
    lmax = 10  # HACK: these are hard-set in Fortran

    pos_path = os.path.abspath(get_pkg_data_filename('data/{}-samples.dat.gz'.format(basename)))
    coeff_path = os.path.abspath(get_pkg_data_filename('data/computed-{0}.coeff'.format(basename)))
    coeff = np.atleast_2d(np.loadtxt(coeff_path))

    xyz = np.ascontiguousarray(np.loadtxt(pos_path, skiprows=1))
    S, T = compute_coeffs_discrete(xyz, mass=np.zeros(xyz.shape[0])+1./xyz.shape[0],
                                   nmax=nmax, lmax=lmax, r_s=1.)

    S_f77 = np.zeros((nmax+1, lmax+1, lmax+1))
    T_f77 = np.zeros((nmax+1, lmax+1, lmax+1))
    for row in coeff:
        n, l, m, cc, sc = row

        # transform from H&O 1992 coefficients to Lowing 2011 coefficients
        if l != 0:
            fac = np.sqrt(4*np.pi) * np.sqrt((2*l+1) / (4*np.pi) * factorial(l-m) / factorial(l+m))
            cc /= fac
            sc /= fac

        S_f77[int(n), int(l), int(m)] = -cc
        T_f77[int(n), int(l), int(m)] = -sc

    assert np.allclose(S_f77, S)
