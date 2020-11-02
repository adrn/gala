# coding: utf-8

import os

# Third-party
from astropy.utils.data import get_pkg_data_filename
from astropy.constants import G as _G
import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np
from scipy.integrate import quad
import pytest

# Project
import gala.potential as gp
from gala.units import galactic
from gala._cconfig import GSL_ENABLED
from ..core import compute_coeffs
from .._bfe import potential, density, gradient

G = _G.decompose(galactic).value

if not GSL_ENABLED:
    pytest.skip("skipping SCF tests: they depend on GSL",
                allow_module_level=True)


# Check that we get A000=1. for putting in hernquist density
def hernquist_density(x, y, z, M, r_s):
    r = np.sqrt(x**2 + y**2 + z**2)
    return M/(2*np.pi) * r_s / (r * (r+r_s)**3)


def test_hernquist():
    for M in [1E5, 1E10]:
        for r_s in np.logspace(-1, 2, 4):
            (S, Serr), (T, Terr) = compute_coeffs(hernquist_density, nmax=0, lmax=0,
                                                  M=M, r_s=r_s, args=(M, r_s))

            np.testing.assert_allclose(S, 1.)
            np.testing.assert_allclose(Serr, 0., atol=1E-10)

            np.testing.assert_allclose(T, 0.)
            np.testing.assert_allclose(Terr, 0., atol=1E-10)


def test_hernquist_spherical():
    (S, Serr), (T, Terr) = compute_coeffs(hernquist_density, nmax=8, lmax=8,
                                          M=1., r_s=1., args=(1., 1.), skip_m=True)

    np.testing.assert_allclose(S[0, 0, 0], 1., atol=1E-13)
    np.testing.assert_allclose(S[1:, :, :], 0., atol=1E-13)
    np.testing.assert_allclose(Serr, 0., atol=1E-10)

    np.testing.assert_allclose(T, 0., atol=1E-13)
    np.testing.assert_allclose(Terr, 0., atol=1E-10)

# ----------------------------------------------------------------------------


def _plummer_density(x, y, z, M, r_s):
    r2 = x*x + y*y + z*z
    return (3*M / (4*np.pi*r_s**3)) * (1 + r2/r_s**2)**(-5/2.)


def test_plummer():
    true_M = 1/G
    true_r_s = 1.

    x = np.logspace(-2, 1, 512)
    xyz = np.zeros((len(x), 3))
    xyz[:, 0] = x

    pot = gp.PlummerPotential(m=true_M, b=true_r_s, units=galactic)
    true_pot = pot.energy(xyz.T).value
    true_dens = pot.density(xyz.T).value
    true_grad = pot.gradient(xyz.T).value.T

    nmax = 16
    lmax = 0

    (S, S_err), (T, T_err) = compute_coeffs(_plummer_density, nmax=nmax, lmax=lmax,
                                            M=true_M, r_s=true_r_s, args=(true_M, true_r_s),
                                            epsrel=1E-9)

    bfe_dens = density(xyz, S, T, true_M, true_r_s)
    bfe_pot = potential(xyz, S, T, G, true_M, true_r_s)
    bfe_grad = gradient(xyz, S, T, G, true_M, true_r_s)

    # fig, axes = pl.subplots(3, 1, figsize=(6, 12), sharex=True)

    # axes[0].loglog(x, true_dens)
    # axes[0].loglog(x, bfe_dens)

    # axes[1].semilogx(x, true_pot)
    # axes[1].semilogx(x, bfe_pot)

    # axes[2].semilogx(x, true_grad[:, 0])
    # axes[2].semilogx(x, bfe_grad[:, 0])

    # pl.show()

    assert np.allclose(true_dens, bfe_dens, rtol=2E-3)
    assert np.allclose(true_pot, bfe_pot, rtol=1E-6)
    assert np.allclose(true_grad[:, 0], bfe_grad[:, 0], rtol=5E-3)
    # print(np.abs((bfe_dens - true_dens) / true_dens).max())
    # print(np.abs((bfe_pot - true_pot) / true_pot).max())
    # print(np.abs((bfe_grad[:, 0] - true_grad[:, 0]) / true_grad[:, 0]).max())


# ----------------------------------------------------------------------------
# Non-spherical, axisymmetric

def flattened_hernquist_density_s(s, M, a, q):
    return M*a / (2*np.pi) / (s * (a + s)**3)


def flattened_hernquist_density(x, y, z, M, a, q):
    s = np.sqrt(x*x + y*y + z*z/(q*q))
    return flattened_hernquist_density_s(s, M, a, q)


def _integrand_helper(tau, xyz, M, a, q):
    x, y, z = xyz
    m = a*np.sqrt((x*x + y*y) / (a*a + tau) + z*z / (q*q + tau))
    return flattened_hernquist_density_s(m, M, a, q) / ((tau+a*a)*np.sqrt(tau+q*q))


def integrand(tau, i, xyz, M, a, q):
    if i in [0, 1]:
        denom = tau + a*a
    elif i == 2:
        denom = tau + q*q
    else:
        raise ValueError("WTF")

    return _integrand_helper(tau, xyz, M, a, q) * xyz[i] / denom


def flattened_hernquist_gradient(x, y, z, G, M, a, q):
    A = 2*np.pi*G*a**2*q
    gx = A * quad(integrand, 0, np.inf, args=(0, (x, y, z), M, a, q), limit=1000)[0]
    gy = A * quad(integrand, 0, np.inf, args=(1, (x, y, z), M, a, q))[0]
    gz = A * quad(integrand, 0, np.inf, args=(2, (x, y, z), M, a, q))[0]

    return np.array([gx, gy, gz])


def test_flattened_hernquist():
    """
    This test compares the coefficients against some computed in the mathematica
    notebook 'flattened-hernquist.nb'. nmax and lmax here must match nmax and lmax
    in that notebook.
    """

    coeff_path = os.path.abspath(get_pkg_data_filename('data/Snlm-mathematica.csv'))

    G = 1.
    M = 1
    a = 1.
    q = 0.9

    # Note: this must be the same as in the mathematica notebook
    nmax = 8
    lmax = 8

    (Snlm, Serr), (Tnlm, Terr) = compute_coeffs(flattened_hernquist_density,
                                                nmax=nmax, lmax=lmax, skip_odd=True, skip_m=True,
                                                M=M, r_s=a, args=(M, a, q))

    for l in range(1, lmax+1, 2):
        for m in range(lmax+1):
            assert Snlm[0, l, m] == 0.

    m_Snl0 = np.loadtxt(coeff_path, delimiter=',')
    m_Snl0 = m_Snl0[:, ::2]  # every other l

    assert np.allclose(Snlm[0, ::2, 0], m_Snl0[0])

    # check that random points match in gradient and density
    np.random.seed(42)
    n_test = 1024
    r = 10.*np.cbrt(np.random.uniform(0.1**3, 1, size=n_test))  # 1 to 10
    t = np.arccos(2*np.random.uniform(size=n_test) - 1)
    ph = np.random.uniform(0, 2*np.pi, size=n_test)
    x = r*np.cos(ph)*np.sin(t)
    y = r*np.sin(ph)*np.sin(t)
    z = r*np.cos(t)
    xyz = np.vstack((x, y, z))

    # confirmed by testing...
    tru_dens = flattened_hernquist_density(xyz[0], xyz[1], xyz[2], M, a, q)
    bfe_dens = density(np.ascontiguousarray(xyz.T), Snlm, Tnlm, M, a)
    assert np.all((np.abs(bfe_dens - tru_dens) / tru_dens) < 0.05)  # <5%

    tru_grad = np.array([flattened_hernquist_gradient(xyz[0, i], xyz[1, i], xyz[2, i], G, M, a, q)
                        for i in range(xyz.shape[1])]).T
    bfe_grad = gradient(np.ascontiguousarray(xyz.T), Snlm, Tnlm, G, M, a).T

    # check what typical errors are
    # for j in range(3):
    #     pl.hist(np.abs((bfe_grad[j]-tru_grad[j])/tru_grad[j]))

    for j in range(3):
        assert np.all(np.abs((bfe_grad[j]-tru_grad[j])/tru_grad[j]) < 0.005)  # 0.5%

    return

    # ------------------------------------------------------------------------
    # plots:

    # coefficients
    fig, ax = pl.subplots(1, 1, figsize=(10, 8))
    n, l = np.mgrid[:nmax+1, :lmax+1]
    c = ax.scatter(n.ravel(), l.ravel(), c=Snlm[:, :, 0].ravel(), s=64,
                   norm=mpl.colors.SymLogNorm(1E-5), cmap='RdBu_r',
                   vmin=-100, vmax=100, linewidths=1., edgecolors='#666666')

    ax.xaxis.set_ticks(np.arange(0, nmax+1, 1))
    ax.yaxis.set_ticks(np.arange(0, lmax+1, 1))

    ax.set_xlim(-0.5, nmax+0.5)
    ax.set_ylim(-0.5, lmax+0.5)

    ax.set_xlabel('$n$')
    ax.set_ylabel('$l$')

    tickloc = np.concatenate((-10.**np.arange(2, -5-1, -1),
                              10.**np.arange(-5, 2+1, 1)))
    fig.colorbar(c, ticks=tickloc, format='%.0e')
    fig.tight_layout()

    # contour plot in r, t at ph=0

    rgrid = np.logspace(-1, 1., 128)
    tgrid = np.linspace(0, np.pi, 128)

    r, t = np.meshgrid(rgrid, tgrid)
    x = r*np.sin(t)
    z = r*np.cos(t)

    _xyz = np.vstack((x.ravel(), np.zeros_like(x.ravel()), z.ravel()))
    bfe_dens = density(np.ascontiguousarray(_xyz.T), Snlm, Tnlm, M, a)
    true_dens = flattened_hernquist_density(_xyz[0], _xyz[1], _xyz[2], M, a, q)

    fig, ax = pl.subplots(1, 1, figsize=(8, 8))

    levels = 10**np.linspace(-4.5, 1, 16)
    ax.contour(np.log10(r), t, true_dens.reshape(x.shape),
               levels=levels, colors='k',
               locator=mpl.ticker.LogLocator(), label='True')
    ax.contour(np.log10(r), t, bfe_dens.reshape(x.shape),
               levels=levels, colors='r',
               locator=mpl.ticker.LogLocator(), label='BFE')

    ax.legend()
    fig.tight_layout()

    pl.show()
