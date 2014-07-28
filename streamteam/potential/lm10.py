# coding: utf-8

""" Potential used in Law & Majewski 2010 """

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
from .builtin import *

__all__ = ['LM10Potential']

class LM10Potential(CompositePotential):

    def __init__(self, m_disk=1E11, a=6.5, b=0.26,
                 m_spher=3.4E10, c=0.7,
                 q1=1.38, q2=1., q3=1.36, phi=(97*u.degree).to(u.radian).value,
                 v_c=np.sqrt(2)*(121.858*u.km/u.s).to(u.kpc/u.Myr).value, r_h=12.):

        usys = (u.kpc, u.M_sun, u.Myr, u.radian)

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(usys=usys,
                                                m=m_disk, a=a, b=b)

        kwargs["bulge"] = HernquistPotential(usys=usys,
                                             m=m_spher, c=c)

        kwargs["halo"] = LogarithmicPotential(usys=usys,
                                              q1=q1, q2=q2, q3=q3,
                                              phi=phi, v_c=v_c, r_h=r_h)
        super(LM10Potential,self).__init__(**kwargs)