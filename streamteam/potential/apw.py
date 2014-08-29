# coding: utf-8

""" Potential used in Price-Whelan et al. (in prep.) TODO """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
from .core import CompositePotential
from .cbuiltin import HernquistPotential, MiyamotoNagaiPotential, LeeSutoNFWPotential

__all__ = ['PW14Potential']

class PW14Potential(CompositePotential):

    def __init__(self, m_disk=6.5E10, a=6.5, b=0.26,
                 m_spher=1.5E10, c=0.7,
                 q1=1.4, q2=1., q3=0.6,
                 v_h=0.57649379854, r_h=20.,
                 phi=0., theta=0., psi=0.):

        # Choice of r_h comes from c ~ 10 for MW, and R_vir ~ 200 kpc
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