# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

__all__ = ["cartesian_to_poincare_polar"]

def cartesian_to_poincare_polar(w):
    r"""
    Convert an array of 6D Cartesian positions to Poincaré
    symplectic polar coordinates. These are similar to cylindrical
    coordinates.

    Parameters
    ----------
    w : array_like
        Input array of 6D Cartesian phase-space positions. Should have
        shape ``(norbits,6)``.

    Returns
    -------
    new_w : :class:`~numpy.ndarray`
        Points represented in 6D Poincaré polar coordinates.

    """

    R = np.sqrt(w[...,0]**2 + w[...,1]**2)
    # phi = np.arctan2(w[...,1], w[...,0])
    phi = np.arctan2(w[...,0], w[...,1])

    vR = (w[...,0]*w[...,0+3] + w[...,1]*w[...,1+3]) / R
    vPhi = w[...,0]*w[...,1+3] - w[...,1]*w[...,0+3]

    # pg. 437, Papaphillipou & Laskar (1996)
    sqrt_2THETA = np.sqrt(np.abs(2*vPhi))
    pp_phi = sqrt_2THETA * np.cos(phi)
    pp_phidot = sqrt_2THETA * np.sin(phi)

    z = w[...,2]
    zdot = w[...,2+3]

    new_w = np.vstack((R.T, pp_phi.T, z.T,
                       vR.T, pp_phidot.T, zdot.T)).T
    return new_w
