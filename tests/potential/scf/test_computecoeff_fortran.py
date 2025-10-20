import os
from math import factorial as _factorial

import numpy as np
import pytest
from astropy.utils.data import get_pkg_data_filename

from gala._cconfig import GSL_ENABLED

from ..core import compute_coeffs_discrete

if not GSL_ENABLED:
    pytest.skip("skipping SCF tests: they depend on GSL", allow_module_level=True)

# Compare coefficients computed with Fortran to Biff


def factorial(x):
    return _factorial(int(x))


@pytest.mark.parametrize("basename", ["hernquist"])
def test_coeff(basename):
    nmax = 6
    lmax = 10  # HACK: these are hard-set in Fortran

    pos_path = os.path.abspath(get_pkg_data_filename(f"data/{basename}-samples.dat.gz"))
    coeff_path = os.path.abspath(
        get_pkg_data_filename(f"data/computed-{basename}.coeff")
    )
    coeff = np.atleast_2d(np.loadtxt(coeff_path))

    xyz = np.ascontiguousarray(np.loadtxt(pos_path, skiprows=1))
    S, _T = compute_coeffs_discrete(
        xyz,
        mass=np.zeros(xyz.shape[0]) + 1.0 / xyz.shape[0],
        nmax=nmax,
        lmax=lmax,
        r_s=1.0,
    )

    S_f77 = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    T_f77 = np.zeros((nmax + 1, lmax + 1, lmax + 1))
    for row in coeff:
        n, l, m, cc, sc = row

        # transform from H&O 1992 coefficients to Lowing 2011 coefficients
        if l != 0:
            fac = np.sqrt(4 * np.pi) * np.sqrt(
                (2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m)
            )
            cc /= fac
            sc /= fac

        S_f77[int(n), int(l), int(m)] = -cc
        T_f77[int(n), int(l), int(m)] = -sc

    assert np.allclose(S_f77, S)
