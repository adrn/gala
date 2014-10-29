# coding: utf-8

""" Potential used for Pal 5 Challenge at the Gaia Challenge 2 """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.constants import G
import numpy as np

# Project
# from .core import CartesianCompositePotential
from .cpotential import CCompositePotential
from .cbuiltin import JaffePotential, MiyamotoNagaiPotential, Pal5AxisymmetricNFWPotential
from ..units import galactic

__all__ = ['GC2Pal5Potential']

# change to CartesianCompositePotential for Pure-Python
class GC2Pal5Potential(CCompositePotential):

    def __init__(self, m_disk=6.5E10, a=6.5, b=0.26,
                 m_spher=2E10, c=0.3,
                 m_halo=1.81194E12, Rh=32.26, qz=0.814,
                 units=galactic):

        # Choice of v_h sets circular velocity at Sun to 220 km/s
        self.units = units

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(units=units,
                                                m=m_disk, a=a, b=b)

        kwargs["bulge"] = JaffePotential(units=units,
                                         m=m_spher, c=c)

        kwargs["halo"] = Pal5AxisymmetricNFWPotential(units=units,
                                                      M=m_halo, Rh=Rh, qz=qz)
        super(GC2Pal5Potential,self).__init__(**kwargs)
        self.c_instance.G = G.decompose(units).value
