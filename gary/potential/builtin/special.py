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
from ..core import CompositePotential
from .cybuiltin import HernquistPotential, MiyamotoNagaiPotential, \
    LeeSutoTriaxialNFWPotential, SphericalNFWPotential, LogarithmicPotential
from ...units import galactic

__all__ = ['PW14Potential', 'TriaxialMWPotential', 'PyLM10Potential']

# TODO: I hacked together value and gradient functions for LM10 but not others
class PyLM10Potential(CompositePotential):

    def __init__(self, units=galactic, disk=dict(), bulge=dict(), halo=dict()):

        default_disk = dict(m=1E11, a=6.5, b=0.26)
        default_bulge = dict(m=3.4E10, c=0.7)
        default_halo = dict(q1=1.38, q2=1., q3=1.36, r_h=12.,
                            phi=(97*u.degree).to(u.radian).value,
                            v_c=np.sqrt(2)*(121.858*u.km/u.s).to(u.kpc/u.Myr).value)

        for k,v in default_disk.items():
            if k not in disk:
                disk[k] = v

        for k,v in default_bulge.items():
            if k not in bulge:
                bulge[k] = v

        for k,v in default_halo.items():
            if k not in halo:
                halo[k] = v

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(units=units, **disk)
        kwargs["bulge"] = HernquistPotential(units=units, **bulge)
        kwargs["halo"] = LogarithmicPotential(units=units, **halo)
        super(PyLM10Potential,self).__init__(**kwargs)

# --------------------------------------------------------------------

class PW14Potential(CompositePotential):

    def __init__(self, units=galactic, disk=dict(), bulge=dict(), halo=dict()):

        default_disk = dict(m=6.5E10, a=6.5, b=0.26)
        default_bulge = dict(m=2E10, c=0.3)
        default_halo = dict(a=1.4, b=1., c=0.6, v_c=0.247, r_s=30.,
                            phi=np.pi/2., theta=np.pi/2., psi=np.pi/2.)

        for k,v in default_disk.items():
            if k not in disk:
                disk[k] = v

        for k,v in default_bulge.items():
            if k not in bulge:
                bulge[k] = v

        for k,v in default_halo.items():
            if k not in halo:
                halo[k] = v

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(units=units, **disk)
        kwargs["bulge"] = HernquistPotential(units=units, **bulge)

        if halo['a'] == 1 and halo['b'] == 1 and halo['c'] == 1:
            kwargs["halo"] = SphericalNFWPotential(units=units,
                                                   v_c=halo['v_c'],
                                                   r_s=halo['r_s'])
        else:
            kwargs["halo"] = LeeSutoTriaxialNFWPotential(units=units, **halo)

        super(PW14Potential,self).__init__(**kwargs)

class TriaxialMWPotential(CompositePotential):

    def __init__(self, units=galactic,
                 disk=dict(), bulge=dict(), halo=dict()):
        """ Axis ratio values taken from Jing & Suto (2002). Other
            parameters come from a by-eye fit to Bovy's MW2014Potential.
            Choice of v_c sets circular velocity at Sun to 220 km/s
        """

        default_disk = dict(m=7E10, a=3.5, b=0.14)
        default_bulge = dict(m=1E10, c=1.1)
        default_halo = dict(a=1., b=0.75, c=0.55,
                            v_c=0.239225, r_s=30.,
                            phi=0., theta=0., psi=0.)

        for k,v in default_disk.items():
            if k not in disk:
                disk[k] = v

        for k,v in default_bulge.items():
            if k not in bulge:
                bulge[k] = v

        for k,v in default_halo.items():
            if k not in halo:
                halo[k] = v

        kwargs = dict()
        kwargs["disk"] = MiyamotoNagaiPotential(units=units, **disk)
        kwargs["bulge"] = HernquistPotential(units=units, **bulge)
        kwargs["halo"] = LeeSutoTriaxialNFWPotential(units=units, **halo)
        super(TriaxialMWPotential,self).__init__(**kwargs)

stuff = """
def busey():
    import webbrowser
    webbrowser.open("http://i.imgur.com/KNoyPwW.jpg")
"""
