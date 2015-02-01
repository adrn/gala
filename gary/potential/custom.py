# coding: utf-8

""" Potential used in Price-Whelan et al. (in prep.) TODO """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.constants import G
import astropy.units as u
import numpy as np

# Project
# from .cpotential import CCompositePotential
from .core import CompositePotential
from .cbuiltin import HernquistPotential, MiyamotoNagaiPotential, \
    LeeSutoTriaxialNFWPotential, SphericalNFWPotential, LogarithmicPotential
from ..units import galactic

__all__ = ['PW14Potential', 'LM10Potential', 'TriaxialMWPotential']

class PW14Potential(CompositePotential):

    def __init__(self, m_disk=6.5E10, a=6.5, b=0.26,
                 m_spher=2E10, c=0.3,
                 q1=1.4, q2=1., q3=0.6,
                 v_c=0.247, r_s=30.,
                 phi=np.pi/2., theta=np.pi/2., psi=np.pi/2.,
                 units=galactic):

        # Choice of v_h sets circular velocity at Sun to 220 km/s
        self.units = units

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(units=units,
                                                m=m_disk, a=a, b=b)

        kwargs["bulge"] = HernquistPotential(units=units,
                                             m=m_spher, c=c)

        if q1 == 1 and q2 == 1 and q3 == 1:
            kwargs["halo"] = SphericalNFWPotential(units=units,
                                                   v_c=v_c, r_s=r_s)
        else:
            kwargs["halo"] = LeeSutoTriaxialNFWPotential(units=units,
                                                         a=q1, b=q2, c=q3,
                                                         v_c=v_c, r_s=r_s,
                                                         phi=phi, theta=theta, psi=psi)
        super(PW14Potential,self).__init__(**kwargs)

class LM10Potential(CompositePotential):

    def __init__(self, m_disk=1E11, a=6.5, b=0.26,
                 m_spher=3.4E10, c=0.7,
                 q1=1.38, q2=1., q3=1.36, phi=(97*u.degree).to(u.radian).value,
                 v_c=np.sqrt(2)*(121.858*u.km/u.s).to(u.kpc/u.Myr).value, r_h=12.,
                 units=galactic):

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(units=units,
                                                m=m_disk, a=a, b=b)

        kwargs["bulge"] = HernquistPotential(units=units,
                                             m=m_spher, c=c)

        kwargs["halo"] = LogarithmicPotential(units=units,
                                              q1=q1, q2=q2, q3=q3,
                                              phi=phi, v_c=v_c, r_h=r_h)
        super(LM10Potential,self).__init__(**kwargs)

class TriaxialMWPotential(CompositePotential):

    def __init__(self, m_disk=7E10, a=3.5, b=0.14,
                 m_spher=1E10, c=1.1,
                 q1=1., q2=0.75, q3=0.55,
                 v_c=0.239225, r_s=30.,
                 phi=0., theta=0., psi=0.,
                 units=galactic):
        """ Axis ratio values taken from Jing & Suto (2002). Other
            parameters come from a by-eye fit to Bovy's MW2014Potential.
        """

        # Choice of v_h sets circular velocity at Sun to 220 km/s
        self.units = units

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(units=units,
                                                m=m_disk, a=a, b=b)

        kwargs["bulge"] = HernquistPotential(units=units,
                                             m=m_spher, c=c)

        kwargs["halo"] = LeeSutoTriaxialNFWPotential(units=units,
                                                     a=q1, b=q2, c=q3,
                                                     v_c=v_c, r_s=r_s,
                                                     phi=phi, theta=theta, psi=psi)
        super(TriaxialMWPotential,self).__init__(**kwargs)
