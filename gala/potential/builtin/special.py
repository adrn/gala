# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard lib
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np

# Project
# from .cpotential import CCompositePotential
# from ..core import CompositePotential
from .cybuiltin import HernquistPotential, MiyamotoNagaiPotential, \
    LeeSutoTriaxialNFWPotential, SphericalNFWPotential, LogarithmicPotential,\
    CCompositePotential
from ...units import galactic

__all__ = ['LM10Potential'] # ['TriaxialMWPotential']

class LM10Potential(CCompositePotential):
    """
    The Galactic potential used by Law and Majewski (2010) to represent
    the Milky Way as a three-component sum of disk, bulge, and halo.

    The disk potential is an axisymmetric
    :class:`~gala.potential.MiyamotoNagaiPotential`, the bulge potential
    is a spherical :class:`~gala.potential.HernquistPotential`, and the
    halo potential is a triaxial :class:`~gala.potential.LogarithmicPotential`.

    Default parameters are fixed to those found in LM10 by fitting N-body
    simulations to the Sagittarius stream.

    Parameters
    ----------
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    disk : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.MiyamotoNagaiPotential`.
    bulge : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.HernquistPotential`.
    halo : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.LogarithmicPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """
    def __init__(self, units=galactic,
                 disk=dict(), bulge=dict(), halo=dict()):

        default_disk = dict(m=1E11*u.Msun, a=6.5*u.kpc, b=0.26*u.kpc)
        default_bulge = dict(m=3.4E10*u.Msun, c=0.7*u.kpc)
        default_halo = dict(q1=1.38, q2=1., q3=1.36, r_h=12.*u.kpc,
                            phi=97*u.degree,
                            v_c=np.sqrt(2)*121.858*u.km/u.s)

        for k,v in default_disk.items():
            if k not in disk:
                disk[k] = v

        for k,v in default_bulge.items():
            if k not in bulge:
                bulge[k] = v

        for k,v in default_halo.items():
            if k not in halo:
                halo[k] = v

        super(LM10Potential,self).__init__()

        self["disk"] = MiyamotoNagaiPotential(units=units, **disk)
        self["bulge"] = HernquistPotential(units=units, **bulge)
        self["halo"] = LogarithmicPotential(units=units, **halo)
        self.lock = True

# --------------------------------------------------------------------
# class TriaxialMWPotential(CCompositePotential):

#     def __init__(self, units=galactic,
#                  disk=dict(), bulge=dict(), halo=dict()):
#         """ Axis ratio values taken from Jing & Suto (2002). Other
#             parameters come from a by-eye fit to Bovy's MW2014Potential.
#             Choice of v_c sets circular velocity at Sun to 220 km/s
#         """

#         default_disk = dict(m=7E10, a=3.5, b=0.14)
#         default_bulge = dict(m=1E10, c=1.1)
#         default_halo = dict(a=1., b=0.75, c=0.55,
#                             v_c=0.239225, r_s=30.,
#                             phi=0., theta=0., psi=0.)

#         for k,v in default_disk.items():
#             if k not in disk:
#                 disk[k] = v

#         for k,v in default_bulge.items():
#             if k not in bulge:
#                 bulge[k] = v

#         for k,v in default_halo.items():
#             if k not in halo:
#                 halo[k] = v

#         kwargs = dict()
#         kwargs["disk"] = MiyamotoNagaiPotential(units=units, **disk)
#         kwargs["bulge"] = HernquistPotential(units=units, **bulge)
#         kwargs["halo"] = LeeSutoTriaxialNFWPotential(units=units, **halo)
#         super(TriaxialMWPotential,self).__init__(**kwargs)
# --------------------------------------------------------------------

stuff = """
def busey():
    import webbrowser
    webbrowser.open("http://i.imgur.com/KNoyPwW.jpg")
"""
