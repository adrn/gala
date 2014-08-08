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
    double pow(double x, double n)

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
    cdef public double v_h, r_h, a, b, c, e_b2, e_c2
    cdef public double v_h2, r_h2, a2, b2, c2, x0

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

        self.e_b2 = 1-pow(b/a,2)
        self.e_c2 = 1-pow(c/a,2)

        self.x0 = 1/r_h

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _value(self, double[:,::1] r,
                                   double[::1] pot, int nparticles):

        cdef double x, y, z, _r, u
        for i in range(nparticles):
            x = r[i,0]
            y = r[i,1]
            z = r[i,2]

            _r = sqrt(x*x + y*y + z*z)
            u = _r / self.r_h
            pot[i] = self.v_h2*(2*sqrt((u + 1)/u) + (self.e_b2/2 + self.e_c2/2)*(sqrt(u/(u + 1))*(-1 - 13/(6*u) + 5/(12*u*u) + 5/(4*u*u*u)) + (2/u - 5/(4*u*u*u))*log(sqrt(u) + sqrt(u + 1)) + 1) + (sqrt(u/(u + 1))*(1/(2*u) - 5/(4*u*u) - 15/(4*u*u*u)) + 15*log(sqrt(u) + sqrt(u + 1))/(4*u*u*u))*(self.e_b2*y**2/(2*_r*_r) + self.e_c2*z*z/(2*_r*_r)) - 2 - 2*log(sqrt(u) + sqrt(u + 1))/u)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public inline void _gradient(self, double[:,::1] r,
                                      double[:,::1] grad, int nparticles):

        cdef:
            double x, y, z, _r
            double x1, x2, x3, x4, x5, x6, x7, x8
            double x10, x11, x12, x13, x16, x17, x18, x19
            double x20, x21, x22, x23, x24, x25, x26, x27, x28, x29
            double x30, x31

        for i in range(nparticles):
            x = r[i,0]
            y = r[i,1]
            z = r[i,2]
            _r = sqrt(x*x + y*y + z*z)

            x1 = _r + self.r_h
            x2 = 1/x1
            x3 = self.x0*x1
            x4 = pow(x3,1.5)
            x5 = sqrt(_r*self.x0)
            x6 = sqrt(x3)
            x7 = x5 + x6
            x8 = self.v_h2*self.x0*x2/(48*pow(_r,7)*x4*x7)
            x10 = 12*x4*x7*self.r_h2
            x11 = _r*_r*_r*_r
            x12 = log(x7)
            x13 = -4*_r*_r*_r*_r*_r*sqrt(x1/_r) + 8*x1*x11*x12
            x16 = self.e_b2*y*y + self.e_c2*z*z
            x17 = 15*self.r_h2
            x18 = sqrt(_r*x2)
            x19 = _r*self.r_h
            x20 = _r*_r
            x21 = -2*x20
            x22 = x18*(x17 + 5*x19 + x21)
            x23 = x1*(-x12*x17 + x22)
            x24 = _r + self.r_h*x5*x6
            x25 = x1*x24
            x26 = self.r_h*x6*x7
            x27 = 2*x1*x18
            x28 = 10*x19 + 45*self.r_h2
            x29 = -x17
            x30 = 26*x20
            x31 = 3*self.r_h*x1*x16*(15*self.r_h*x25 - 90*x1*x12*x6*x7*self.r_h2 - x22*x26 + x27*x6*x7*(x21 + x28)) - 48*x1**2*x11*x24 - x1*x20*(self.e_b2 + self.e_c2)*(self.r_h*x27*x6*x7*(x28 - x30) + 6*x1*x12*x26*(8*x20 + x29) + x18*x26*(12*_r*_r*_r - 5*_r*self.r_h2 - 15*self.r_h2*self.r_h + self.r_h*x30) - x25*(24*x20 + x29))

            grad[i,0] = x*x8*(x10*(x13 + x16*x23) + x31)
            grad[i,1] = x8*y*(x10*(x13 + x23*(-self.e_b2*x20 + x16)) + x31)
            grad[i,2] = x8*z*(x10*(x13 + x23*(-self.e_c2*x20 + x16)) + x31)

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