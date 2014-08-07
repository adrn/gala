# coding: utf-8

""" Built-in potentials implemented in Cython """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
from astropy.constants import G
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

# Project
from .cpotential cimport _CPotential
from .cpotential import CPotential
from .core import CartesianPotential

cdef extern from "math.h":
    double sqrt(double x)
    double atan2(double x, double x)
    double acos(double x)
    double sin(double x)
    double cos(double x)
    double log(double x)
    double fabs(double x)
    double exp(double x)

__all__ = ['MiyamotoNagaiPotential']

##############################################################################
#    Miyamoto-Nagai Disk potential from Miyamoto & Nagai 1975
#    http://adsabs.harvard.edu/abs/1975PASJ...27..533M
#
cdef class _MiyamotoNagaiPotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double G, GM
    cdef public double m, a, b, b2

    def __init__(self, double G, double m, double a, double b):
        """ Units of everything should be in the system:
                kpc, Myr, radian, M_sun
        """
        # cdef double G = 4.499753324353494927e-12 # kpc^3 / Myr^2 / M_sun
        # have to specify G in the correct units
        self.G = G

        # disk parameters
        self.GM = G*m
        self.m = m
        self.a = a
        self.b = b
        self.b2 = b*b

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _value(self, double[:,::1] r,
                                   double[::1] pot, int nparticles):

        cdef double x, y, z
        cdef double zd
        for i in range(nparticles):
            x = r[i,0]
            y = r[i,1]
            z = r[i,2]

            zd = (self.a + sqrt(z*z+self.b2))
            pot[i] = -self.GM / np.sqrt(x*x + y*y + zd*zd)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _gradient(self, double[:,::1] r,
                                      double[:,::1] grad, int nparticles):

        cdef double x, y, z
        cdef double sqrtz, zd, fac
        for i in range(nparticles):
            x = r[i,0]
            y = r[i,1]
            z = r[i,2]

            sqrtz = sqrt(z*z + self.b2)
            zd = self.a + sqrtz
            fac = self.GM*pow(x*x + y*y + zd*zd, -1.5)

            grad[i,0] = fac*x
            grad[i,1] = fac*y
            grad[i,2] = fac*z * (1. + self.a / sqrtz)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _hessian(self, double[:,::1] r,
                                     double[:,:,::1] hess, int nparticles):

        cdef double x, y, z
        cdef double sqrtz, zd, fac
        for i in range(nparticles):
            x = r[i,0]
            y = r[i,1]
            z = r[i,2]

            sqrtz = sqrt(z*z + self.b2)
            zd = self.a + sqrtz
            fac = self.GM*pow(x*x + y*y + zd*zd, -1.5)

            grad[i,0] = fac*x
            grad[i,1] = fac*y
            grad[i,2] = fac*z * (1. + self.a / sqrtz)


class MiyamotoNagaiPotential(CPotential, CartesianPotential):
    r"""
    Miyamoto-Nagai potential for a flattened mass distribution.

    .. math::

        \Phi_{disk} = -\frac{GM_{disk}}{\sqrt{R^2 + (a + \sqrt{z^2 + b^2})^2}}

    Parameters
    ----------
    m : numeric
        Mass.
    a : numeric
    b : numeric
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, a, b, usys):
        self.usys = usys
        _G = G.decompose(usys).value
        parameters = dict(G=_G, m=m, a=a, b=b)
        super(MiyamotoNagaiPotential, self).__init__(_MiyamotoNagaiPotential,
                                                     parameters=parameters)