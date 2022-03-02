# coding: utf-8

# Standard library
import os
from math import factorial as _factorial

# Third-party
from astropy.constants import G as _G
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import pytest

# Project
from gala.units import galactic
from gala._cconfig import GSL_ENABLED
from .._bfe import density, potential, gradient

G = _G.decompose(galactic).value

if not GSL_ENABLED:
    pytest.skip("skipping SCF tests: they depend on GSL",
                allow_module_level=True)


def factorial(x):
    return _factorial(int(x))


@pytest.mark.parametrize("basename", [
    'simple-hernquist', 'multi-hernquist', 'simple-nonsph', 'random', 'wang-zhao',
])
def test_density(basename):
    pos_path = os.path.abspath(get_pkg_data_filename('data/positions.dat.gz'))
    coeff_path = os.path.abspath(
        get_pkg_data_filename(f'data/{basename}.coeff'))
    accp_path = os.path.abspath(
        get_pkg_data_filename(f'data/{basename}-accp.dat.gz'))

    xyz = np.ascontiguousarray(np.loadtxt(pos_path, skiprows=1).T)
    coeff = np.atleast_2d(np.loadtxt(coeff_path, skiprows=1))

    nmax = coeff[:, 0].astype(int).max()
    lmax = coeff[:, 1].astype(int).max()

    cos_coeff = np.zeros((nmax+1, lmax+1, lmax+1))
    sin_coeff = np.zeros((nmax+1, lmax+1, lmax+1))
    for row in coeff:
        n, l, m, cc, sc = row

        # transform from H&O 1992 coefficients to Lowing 2011 coefficients
        if l != 0:
            fac = np.sqrt(4*np.pi) * np.sqrt((2*l+1) / (4*np.pi) * factorial(l-m) / factorial(l+m))
            cc /= fac
            sc /= fac

        cos_coeff[int(n), int(l), int(m)] = cc
        sin_coeff[int(n), int(l), int(m)] = sc

    dens = density(xyz, M=1., r_s=1.,
                   Snlm=cos_coeff, Tnlm=sin_coeff)

    # TODO: nothing to compare this to....
    # just test that it runs...


@pytest.mark.parametrize("basename", [
    'simple-hernquist', 'multi-hernquist', 'simple-nonsph', 'random', 'wang-zhao',
])
def test_potential(basename):
    coeff_path = os.path.abspath(get_pkg_data_filename('data/{0}.coeff'.format(basename)))
    accp_path = os.path.abspath(get_pkg_data_filename('data/{0}-accp.dat.gz'.format(basename)))

    coeff = np.atleast_2d(np.loadtxt(coeff_path, skiprows=1))
    accp = np.loadtxt(accp_path)

    pos_path = os.path.abspath(get_pkg_data_filename('data/positions.dat.gz'))
    xyz = np.loadtxt(pos_path, skiprows=1)

    nmax = coeff[:, 0].astype(int).max()
    lmax = coeff[:, 1].astype(int).max()

    cos_coeff = np.zeros((nmax+1, lmax+1, lmax+1))
    sin_coeff = np.zeros((nmax+1, lmax+1, lmax+1))
    for row in coeff:
        n, l, m, cc, sc = row

        # transform from H&O 1992 coefficients to Lowing 2011 coefficients
        if l != 0:
            fac = np.sqrt(4*np.pi) * np.sqrt((2*l+1) / (4*np.pi) * factorial(l-m) / factorial(l+m))
            cc /= fac
            sc /= fac

        cos_coeff[int(n), int(l), int(m)] = cc
        sin_coeff[int(n), int(l), int(m)] = sc

    potv = potential(xyz, G=1., M=1., r_s=1.,
                     Snlm=cos_coeff, Tnlm=sin_coeff)

    # for some reason, SCF potential is -potential
    scf_potv = -accp[:, -1]
    np.testing.assert_allclose(potv, scf_potv, rtol=1E-6)


@pytest.mark.parametrize("basename", [
    'simple-hernquist', 'multi-hernquist', 'simple-nonsph', 'random', 'wang-zhao',
])
def test_gradient(basename):
    pos_path = os.path.abspath(get_pkg_data_filename('data/positions.dat.gz'))
    coeff_path = os.path.abspath(get_pkg_data_filename('data/{0}.coeff'.format(basename)))
    accp_path = os.path.abspath(get_pkg_data_filename('data/{0}-accp.dat.gz'.format(basename)))

    xyz = np.loadtxt(pos_path, skiprows=1)
    coeff = np.atleast_2d(np.loadtxt(coeff_path, skiprows=1))
    accp = np.loadtxt(accp_path)

    nmax = coeff[:, 0].astype(int).max()
    lmax = coeff[:, 1].astype(int).max()

    cos_coeff = np.zeros((nmax+1, lmax+1, lmax+1))
    sin_coeff = np.zeros((nmax+1, lmax+1, lmax+1))
    for row in coeff:
        n, l, m, cc, sc = row

        # transform from H&O 1992 coefficients to Lowing 2011 coefficients
        if l != 0:
            fac = np.sqrt(4*np.pi) * np.sqrt((2*l+1) / (4*np.pi) * factorial(l-m) / factorial(l+m))
            cc /= fac
            sc /= fac

        cos_coeff[int(n), int(l), int(m)] = cc
        sin_coeff[int(n), int(l), int(m)] = sc

    grad = gradient(xyz, G=1., M=1., r_s=1.,
                    Snlm=cos_coeff, Tnlm=sin_coeff)

    # I output the acceleration from SCF when I make the files
    #   so I have no idea why I don't need a minus sign here...
    scf_grad = accp[:, :3]
    np.testing.assert_allclose(grad, scf_grad, rtol=1E-6)
