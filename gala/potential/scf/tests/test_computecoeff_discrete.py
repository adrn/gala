# coding: utf-8

from __future__ import division, print_function


import os

# Third-party
import numpy as np
from astropy.utils.data import get_pkg_data_filename
from astropy.constants import G
import gala.potential as gp
from gala.units import galactic
_G = G.decompose(galactic).value

# Project
from ..core import compute_coeffs_discrete
from .._bfe import potential

def test_plummer():
    pos_path = os.path.abspath(get_pkg_data_filename('data/plummer-pos.dat.gz'))

    scfbi = scfbi = np.loadtxt(pos_path)
    m_k = scfbi[:,0]*10 # masses sum to 0.1
    xyz = scfbi[:,1:4]

    G = 1.
    r_s = 1.
    M = m_k.sum()
    pot = gp.PlummerPotential(m=1/_G, b=r_s, units=galactic)

    nmax = 10
    lmax = 0

    Snlm,Tnlm = compute_coeffs_discrete(xyz, m_k, nmax=nmax, lmax=lmax, r_s=r_s)

    x = np.logspace(-2,1,512)
    xyz = np.zeros((len(x),3))
    xyz[:,0] = x

    # plot discrete vs. analytic potential
    true_pot = pot.value(xyz.T).value
    bfe_pot = potential(xyz, Snlm, Tnlm, G, M, r_s)

    # import matplotlib.pyplot as pl

    # pl.figure()
    # pl.semilogx(x, true_pot, marker='.', ls='none')
    # pl.semilogx(x, bfe_pot, marker=None)
    # # pl.semilogx(x, (true_pot-bfe_pot)/true_pot, marker=None)
    # pl.show()
    # # return

    assert np.allclose(true_pot, bfe_pot, rtol=1E-2)
