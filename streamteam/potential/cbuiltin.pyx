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

__all__ = ['MiyamotoNagaiPotential', 'LeeSutoNFWPotential']

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

    # @cython.boundscheck(False)
    # @cython.cdivision(True)
    # @cython.wraparound(False)
    # @cython.nonecheck(False)
    # cdef public inline void _hessian(self, double[:,::1] r,
    #                                  double[:,:,::1] hess, int nparticles):

    #     cdef double x, y, z
    #     cdef double sqrtz, zd, fac
    #     for i in range(nparticles):
    #         x = r[i,0]
    #         y = r[i,1]
    #         z = r[i,2]

    #         sqrtz = sqrt(z*z + self.b2)
    #         zd = self.a + sqrtz
    #         fac = self.GM*pow(x*x + y*y + zd*zd, -1.5)

    #         grad[i,0] = fac*x
    #         grad[i,1] = fac*y
    #         grad[i,2] = fac*z * (1. + self.a / sqrtz)


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

##############################################################################
#    Lee & Suto (2003) triaxial NFW potential
#    http://adsabs.harvard.edu/abs/2003ApJ...585..151L
#
cdef class _LeeSutoNFWPotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double v_h, r_h, a, b, c
    cdef public double v_h2, r_h2, a2, b2, c2, x0, x16, x18, x20

    def __init__(self, double v_h, double r_h, double a, double b, double c):
        """ Units of everything should be in the system:
                kpc, Myr, radian, M_sun
        """

        self.v_h = v_h
        self.v_h2 = v_h*v_h
        self.r_h = r_h
        self.r_h2 = r_h*r_h
        self.a = a
        self.a2 = a*a
        self.b = b
        self.b2 = b*b
        self.c = c
        self.c2 = c*c

        self.x0 = 1/r_h
        self.x16 = self.a2 - self.b2
        self.x18 = self.a2 - self.c2
        self.x20 = 15*self.r_h2

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _value(self, double[:,::1] r,
                                   double[::1] pot, int nparticles):

        cdef double x, y, z, _r
        for i in range(nparticles):
            x = r[i,0]
            y = r[i,1]
            z = r[i,2]

            _r = sqrt(x*x + y*y + z*z)
            pot[i] = -self.v_h2*(48*self.r_h*self.a2*_r*_r*_r*_r*log(sqrt(_r/self.r_h) + sqrt((self.r_h + _r)/self.r_h)) - 3*self.r_h*(15*self.r_h**2*log(sqrt(_r/self.r_h) + sqrt((self.r_h + _r)/self.r_h)) - sqrt(_r/(self.r_h + _r))*(15*self.r_h*self.r_h + 5*self.r_h*_r - 2*_r*_r))*(y*y*(self.a2 - self.b2) + z*z*(self.a2 - self.c2)) - 48*self.a2*_r*_r*_r*_r*_r*(sqrt((self.r_h + _r)/_r) - 1) + _r*_r*(-2*self.a2 + self.b2 + self.c2)*(-3*self.r_h*(5*self.r_h*self.r_h - 8*_r*_r)*log(sqrt(_r/self.r_h) + sqrt((self.r_h + _r)/self.r_h)) + 12*_r*_r*_r + sqrt(_r/(self.r_h + _r))*(15*self.r_h*self.r_h*self.r_h + 5*self.r_h*self.r_h*_r - 26*self.r_h*_r*_r - 12*_r*_r*_r)))/(24*self.a2*_r*_r*_r*_r*_r)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _gradient(self, double[:,::1] r,
                                      double[:,::1] grad, int nparticles):

        cdef:
            double x, y, z, _r
            double x2, x3, x4, x5, x6, x7, x8, x9
            double x11, x12, x13, x14, x19
            double x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33

        for i in range(nparticles):
            x = r[i,0]
            y = r[i,1]
            z = r[i,2]
            _r = sqrt(x*x + y*y + z*z)

            x2 = self.r_h + _r
            x3 = 1/x2
            x4 = self.x0*x2
            x5 = x4**(3/2)
            x6 = sqrt(_r*self.x0)
            x7 = sqrt(x4)
            x8 = x6 + x7
            x9 = self.v_h2*self.x0*x3/(48*_r**7*self.a2*x5*x8)
            x11 = 12*self.r_h2*x5*x8
            x12 = _r*_r*_r*_r
            x13 = log(x8)
            x14 = 4*_r**5*self.a2*sqrt(x2/_r) - 8*self.a2*x12*x13*x2
            x19 = self.x16*y**2 + self.x18*z**2
            x21 = sqrt(_r*x3)
            x22 = self.r_h*_r
            x23 = _r*_r
            x24 = -2*x23
            x25 = x21*(self.x20 + 5*x22 + x24)
            x26 = x2*(x13*self.x20 - x25)
            x27 = self.r_h*x6*x7 + _r
            x28 = x2*x27
            x29 = self.r_h*x7*x8
            x30 = 2*x2*x21
            x31 = 45*self.r_h2 + 10*x22
            x32 = 26*x23
            x33 = 3*self.r_h*x19*x2*(-15*self.r_h*x28 + 90*self.r_h2*x13*x2*x7*x8 + x25*x29 - x30*x7*x8*(x24 + x31)) + 48*self.a2*x12*x2**2*x27 - x2*x23*(-2*self.a2 + self.b2 + self.c2)*(self.r_h*x30*x7*x8*(x31 - x32) - 6*x13*x2*x29*(self.x20 - 8*x23) - x21*x29*(15*self.r_h**3 - self.r_h*x32 - 12*_r**3 + 5*_r*self.r_h2) + x28*(self.x20 - 24*x23))

            grad[i,0] = -x*x9*(x11*(x14 + x19*x26) + x33)
            grad[i,1] = -x9*y*(x11*(x14 + x26*(-self.x16*x23 + x19)) + x33)
            grad[i,2] = -x9*z*(x11*(x14 + x26*(-self.x18*x23 + x19)) + x33)

class LeeSutoNFWPotential(CPotential, CartesianPotential):
    r"""
    TODO:

    .. math::

        \Phi() =

    Parameters
    ----------
    TODO
    usys : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_h, r_h, a, b, c, usys=None):
        self.usys = usys
        parameters = dict(v_h=v_h, r_h=r_h, a=a, b=b, c=c)
        super(LeeSutoNFWPotential, self).__init__(_LeeSutoNFWPotential,
                                                  parameters=parameters)