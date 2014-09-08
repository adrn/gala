# coding: utf-8

""" Potential used in Price-Whelan et al. (in prep.) TODO """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np

# Project
from .core import CartesianCompositePotential
from .cbuiltin import HernquistPotential, MiyamotoNagaiPotential, LeeSutoNFWPotential

__all__ = ['PW14Potential']


class PW14Potential(CartesianCompositePotential):

    def __init__(self, m_disk=6.5E10, a=6.5, b=0.26,
                 m_spher=2E10, c=0.3,
                 q1=1.4, q2=1., q3=0.6,
                 v_h=0.562, r_h=30.,
                 phi=np.pi/2., theta=np.pi/2., psi=np.pi/2.):

        # Choice of v_h sets circular velocity at Sun to 220 km/s
        usys = (u.kpc, u.M_sun, u.Myr, u.radian)
        self.usys = usys

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(usys=usys,
                                                m=m_disk, a=a, b=b)

        kwargs["bulge"] = HernquistPotential(usys=usys,
                                             m=m_spher, c=c)

        kwargs["halo"] = LeeSutoNFWPotential(usys=usys,
                                             a=q1, b=q2, c=q3,
                                             v_h=v_h, r_h=r_h,
                                             phi=phi, theta=theta, psi=psi)
        super(PW14Potential,self).__init__(**kwargs)
