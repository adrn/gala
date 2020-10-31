# coding: utf-8

# Third-party
import astropy.units as u
from astropy.constants import G as _G
import numpy as np
import pytest

# Project
from gala._cconfig import GSL_ENABLED
from gala.units import galactic
from .._bfe import density, potential, gradient

G = _G.decompose(galactic).value

if not GSL_ENABLED:
    pytest.skip("skipping SCF tests: they depend on GSL",
                allow_module_level=True)


# Check that we get A000=1. for putting in hernquist density
def hernquist_density(xyz, M, r_s):
    r = np.sqrt(np.sum(xyz**2, axis=0))
    return M/(2*np.pi) * r_s / (r * (r+r_s)**3)


def hernquist_potential(xyz, M, r_s):
    r = np.sqrt(np.sum(xyz**2, axis=0))
    return -G*M / (r + r_s)


def hernquist_gradient(xyz, M, r_s):
    import gala.potential as gp
    p = gp.HernquistPotential(m=M, c=r_s,
                              units=[u.kpc, u.Myr, u.Msun, u.radian])
    return p.gradient(xyz).value


def test_hernquist():
    nmax = 6
    lmax = 2

    Snlm = np.zeros((nmax+1, lmax+1, lmax+1))
    Tnlm = np.zeros((nmax+1, lmax+1, lmax+1))
    Snlm[0, 0, 0] = 1.

    M = 1E10
    r_s = 3.5

    nbins = 128
    rr = np.linspace(0.1, 10., nbins)
    xyz = np.zeros((nbins, 3))
    xyz[:, 0] = rr * np.cos(np.pi/4.) * np.sin(np.pi/4.)
    xyz[:, 1] = rr * np.sin(np.pi/4.) * np.sin(np.pi/4.)
    xyz[:, 2] = rr * np.cos(np.pi/4.)

    bfe_dens = density(xyz, Snlm, Tnlm, M=M, r_s=r_s)
    true_dens = hernquist_density(xyz.T, M, r_s)
    np.testing.assert_allclose(bfe_dens, true_dens)

    bfe_pot = potential(xyz, Snlm, Tnlm, G=G, M=M, r_s=r_s)
    true_pot = hernquist_potential(xyz.T, M, r_s)
    np.testing.assert_allclose(bfe_pot, true_pot)

    bfe_grad = gradient(xyz, Snlm, Tnlm, G=G, M=M, r_s=r_s)
    true_grad = hernquist_gradient(xyz.T, M, r_s)
    np.testing.assert_allclose(bfe_grad.T, true_grad)


def pure_py(xyz, Snlm, Tnlm, nmax, lmax):
    from scipy.special import lpmv, gegenbauer, eval_gegenbauer, gamma
    from math import factorial as f

    def Plm(l, m, costh):
        return lpmv(m, l, costh)

    def Ylmth(l, m, costh):
        return np.sqrt((2*l+1)/(4 * np.pi) * f(l-m)/f(l+m)) * Plm(l, m, costh)

    twopi = 2*np.pi
    sqrtpi = np.sqrt(np.pi)
    sqrt4pi = np.sqrt(4*np.pi)

    r = np.sqrt(np.sum(xyz**2, axis=0))
    X = xyz[2] / r  # cos(theta)
    sinth = np.sqrt(1 - X**2)
    phi = np.arctan2(xyz[1], xyz[0])
    xsi = (r - 1) / (r + 1)

    density = 0
    potenti = 0
    gradien = np.zeros_like(xyz)
    sph_gradien = np.zeros_like(xyz)
    for l in range(lmax+1):
        r_term1 = r**l / (r*(1+r)**(2*l+3))
        r_term2 = r**l / (1+r)**(2*l+1)
        for m in range(l+1):
            for n in range(nmax+1):
                Cn = gegenbauer(n, 2*l+3/2)
                Knl = 0.5 * n * (n+4*l+3) + (l+1)*(2*l+1)
                rho_nl = Knl / twopi * sqrt4pi * r_term1 * Cn(xsi)
                phi_nl = -sqrt4pi * r_term2 * Cn(xsi)

                density += rho_nl * Ylmth(l, m, X) * (Snlm[n, l, m]*np.cos(m*phi) +
                                                      Tnlm[n, l, m]*np.sin(m*phi))
                potenti += phi_nl * Ylmth(l, m, X) * (Snlm[n, l, m]*np.cos(m*phi) +
                                                      Tnlm[n, l, m]*np.sin(m*phi))

                # derivatives
                dphinl_dr = (
                    2*sqrtpi*np.power(r, -1 + l)*np.power(1 + r, -3 - 2*l) *
                    (-2*(3 + 4*l)*r*eval_gegenbauer(-1 + n, 2.5 + 2*l, (-1 + r)/(1 + r)) +
                     (1 + r)*(l*(-1 + r) + r)*eval_gegenbauer(n, 1.5 + 2*l, (-1 + r)/(1 + r))))
                sph_gradien[0] += dphinl_dr * Ylmth(l, m, X) * (Snlm[n, l, m]*np.cos(m*phi) +
                                                                Tnlm[n, l, m]*np.sin(m*phi))

                A = np.sqrt((2*l+1) / (4*np.pi)) * np.sqrt(gamma(l-m+1) / gamma(l+m+1))
                dYlm_dth = A / sinth * (l*X*Plm(l, m, X) - (l+m)*Plm(l-1, m, X))
                sph_gradien[1] += (1/r) * dYlm_dth * phi_nl * (Snlm[n, l, m]*np.cos(m*phi) +
                                                               Tnlm[n, l, m]*np.sin(m*phi))

                sph_gradien[2] += (m/(r*sinth)) * phi_nl * Ylmth(l, m, X) * (
                    -Snlm[n, l, m]*np.sin(m*phi) + Tnlm[n, l, m]*np.cos(m*phi))

    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    gradien[0] = sinth*cosphi*sph_gradien[0] + X*cosphi*sph_gradien[1] - sinphi*sph_gradien[2]
    gradien[1] = sinth*sinphi*sph_gradien[0] + X*sinphi*sph_gradien[1] + cosphi*sph_gradien[2]
    gradien[2] = X*sph_gradien[0] - sinth*sph_gradien[1]

    return density, potenti, gradien


def test_pure_py():

    nmax = 6
    lmax = 4

    # xyz = np.array([[1., 0., 1.],
    #                 [1., 1., 0.],
    #                 [0., 1., 1.]])
    xyz = np.random.uniform(-2., 2., size=(128, 3))

    # first try spherical:
    Snlm = np.zeros((nmax+1, lmax+1, lmax+1))
    Snlm[:, 0, 0] = np.logspace(0., -4, nmax+1)
    Tnlm = np.zeros_like(Snlm)

    py_den, py_pot, py_grd = pure_py(xyz.T, Snlm, Tnlm, nmax, lmax)

    cy_den = density(xyz, Snlm, Tnlm, M=1., r_s=1.)
    cy_pot = potential(xyz, Snlm, Tnlm, G=1., M=1., r_s=1.)
    cy_grd = gradient(xyz, Snlm, Tnlm, G=1., M=1., r_s=1.).T

    assert np.allclose(py_den, cy_den)
    assert np.allclose(py_pot, cy_pot)
    assert np.allclose(py_grd, cy_grd)

    # non-spherical:
    Snlm = np.random.uniform(-1, 1, size=(nmax+1, lmax+1, lmax+1))
    Tnlm = np.zeros_like(Snlm)

    py_den, py_pot, py_grd = pure_py(xyz.T, Snlm, Tnlm, nmax, lmax)

    cy_den = density(xyz, Snlm, Tnlm, M=1., r_s=1.)
    cy_pot = potential(xyz, Snlm, Tnlm, G=1., M=1., r_s=1.)
    cy_grd = gradient(xyz, Snlm, Tnlm, G=1., M=1., r_s=1.).T

    assert np.allclose(py_den, cy_den)
    assert np.allclose(py_pot, cy_pot)
    assert np.allclose(py_grd, cy_grd)
