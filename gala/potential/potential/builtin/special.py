# Third-party
import astropy.units as u
import numpy as np

# Project
# from .cpotential import CCompositePotential
# from ..core import CompositePotential
from .cybuiltin import (HernquistPotential,
                        MiyamotoNagaiPotential,
                        LogarithmicPotential,
                        NFWPotential,
                        PowerLawCutoffPotential)
from ..ccompositepotential import CCompositePotential
from ....units import galactic

__all__ = ['LM10Potential', 'MilkyWayPotential', 'BovyMWPotential2014']


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

        for k, v in default_disk.items():
            if k not in disk:
                disk[k] = v

        for k, v in default_bulge.items():
            if k not in bulge:
                bulge[k] = v

        for k, v in default_halo.items():
            if k not in halo:
                halo[k] = v

        super().__init__()

        self["disk"] = MiyamotoNagaiPotential(units=units, **disk)
        self["bulge"] = HernquistPotential(units=units, **bulge)
        self["halo"] = LogarithmicPotential(units=units, **halo)
        self.lock = True


class MilkyWayPotential(CCompositePotential):
    """
    A simple mass-model for the Milky Way consisting of a spherical nucleus and
    bulge, a Miyamoto-Nagai disk, and a spherical NFW dark matter halo.

    The disk model is taken from `Bovy (2015)
    <https://ui.adsabs.harvard.edu/#abs/2015ApJS..216...29B/abstract>`_ - if you
    use this potential, please also cite that work.

    Default parameters are fixed by fitting to a compilation of recent mass
    measurements of the Milky Way, from 10 pc to ~150 kpc.

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
        Parameters to be passed to the :class:`~gala.potential.NFWPotential`.
    nucleus : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.HernquistPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """
    def __init__(self, units=galactic,
                 disk=None, halo=None, bulge=None, nucleus=None):

        default_disk = dict(m=6.8E10*u.Msun, a=3.*u.kpc, b=0.28*u.kpc)
        default_bulge = dict(m=5E9*u.Msun, c=1.0*u.kpc)
        default_nucl = dict(m=1.71E9*u.Msun, c=0.07*u.kpc)
        default_halo = dict(m=5.4E11*u.Msun, r_s=15.62*u.kpc)

        if disk is None:
            disk = dict()

        if halo is None:
            halo = dict()

        if bulge is None:
            bulge = dict()

        if nucleus is None:
            nucleus = dict()

        for k, v in default_disk.items():
            if k not in disk:
                disk[k] = v

        for k, v in default_bulge.items():
            if k not in bulge:
                bulge[k] = v

        for k, v in default_halo.items():
            if k not in halo:
                halo[k] = v

        for k, v in default_nucl.items():
            if k not in nucleus:
                nucleus[k] = v

        super().__init__()

        self["disk"] = MiyamotoNagaiPotential(units=units, **disk)
        self["bulge"] = HernquistPotential(units=units, **bulge)
        self["nucleus"] = HernquistPotential(units=units, **nucleus)
        self["halo"] = NFWPotential(units=units, **halo)
        self.lock = True


class BovyMWPotential2014(CCompositePotential):
    """
    An implementation of the ``MWPotential2014``
    `from galpy <https://galpy.readthedocs.io/en/latest/potential.html>`_
    and described in `Bovy (2015)
    <https://ui.adsabs.harvard.edu/#abs/2015ApJS..216...29B/abstract>`_.

    This potential consists of a spherical bulge and dark matter halo, and a
    Miyamoto-Nagai disk component.

    .. note::

        Because it internally uses the PowerLawCutoffPotential,
        this potential requires GSL to be installed, and Gala must have been
        built and installed with GSL support enaled (the default behavior).
        See http://gala.adrian.pw/en/latest/install.html for more information.

    Parameters
    ----------
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    disk : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.MiyamotoNagaiPotential`.
    bulge : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.PowerLawCutoffPotential`.
    halo : dict (optional)
        Parameters to be passed to the :class:`~gala.potential.NFWPotential`.

    Note: in subclassing, order of arguments must match order of potential
    components added at bottom of init.
    """
    def __init__(self, units=galactic,
                 disk=None, halo=None, bulge=None):

        default_disk = dict(m=68193902782.346756*u.Msun, a=3.*u.kpc, b=280*u.pc)
        default_bulge = dict(m=4501365375.06545*u.Msun, alpha=1.8, r_c=1.9*u.kpc)
        default_halo = dict(m=4.3683325e11*u.Msun, r_s=16*u.kpc)

        if disk is None:
            disk = dict()

        if halo is None:
            halo = dict()

        if bulge is None:
            bulge = dict()

        for k, v in default_disk.items():
            if k not in disk:
                disk[k] = v

        for k, v in default_bulge.items():
            if k not in bulge:
                bulge[k] = v

        for k, v in default_halo.items():
            if k not in halo:
                halo[k] = v

        super().__init__()

        self["disk"] = MiyamotoNagaiPotential(units=units, **disk)
        self["bulge"] = PowerLawCutoffPotential(units=units, **bulge)
        self["halo"] = NFWPotential(units=units, **halo)
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

#         for k, v in default_disk.items():
#             if k not in disk:
#                 disk[k] = v

#         for k, v in default_bulge.items():
#             if k not in bulge:
#                 bulge[k] = v

#         for k, v in default_halo.items():
#             if k not in halo:
#                 halo[k] = v

#         kwargs = dict()
#         kwargs["disk"] = MiyamotoNagaiPotential(units=units, **disk)
#         kwargs["bulge"] = HernquistPotential(units=units, **bulge)
#         kwargs["halo"] = LeeSutoTriaxialNFWPotential(units=units, **halo)
#         super(TriaxialMWPotential, self).__init__(**kwargs)
# --------------------------------------------------------------------
