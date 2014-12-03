# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Built-in potentials implemented in Cython """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
from astropy.coordinates.angles import rotation_matrix
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
    double sqrt(double x) nogil
    double cbrt(double x) nogil
    double sin(double x) nogil
    double cos(double x) nogil
    double log(double x) nogil
    double fabs(double x) nogil
    double exp(double x) nogil
    double pow(double x, double n) nogil

__all__ = ['HernquistPotential', 'PlummerPotential', 'MiyamotoNagaiPotential',
           'SphericalNFWPotential', 'LeeSutoTriaxialNFWPotential', 'LogarithmicPotential',
           'JaffePotential']

# ============================================================================
#    Hernquist Spheroid potential from Hernquist 1990
#    http://adsabs.harvard.edu/abs/1990ApJ...356..359H
#
cdef class _HernquistPotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double G, GM
    cdef public double m, c

    def __init__(self, double G, double m, double c):

        # have to specify G in the correct units
        self.G = G

        # disk parameters
        self.GM = G*m
        self.m = m
        self.c = c

    def __reduce__(self):
        args = (self.G, self.m, self.c)
        return (_HernquistPotential, args)

    cdef public inline double _value(self, double *r) nogil:
        cdef double R
        R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
        return -self.GM / (R + self.c)

    cdef public inline void _gradient(self, double *r, double *grad) nogil:
        cdef double R, fac
        R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
        fac = self.GM / ((R + self.c) * (R + self.c) * R)

        grad[0] += fac*r[0]
        grad[1] += fac*r[1]
        grad[2] += fac*r[2]

class HernquistPotential(CPotential, CartesianPotential):
    r"""
    HernquistPotential(m, c, units)

    Hernquist potential for a spheroid.

    .. math::

        \Phi(r) = -\frac{G M}{r + c}

    Parameters
    ----------
    m : numeric
        Mass.
    c : numeric
        Core concentration.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, c, units):
        self.units = units
        _G = G.decompose(units).value
        parameters = dict(G=_G, m=m, c=c)
        super(HernquistPotential, self).__init__(_HernquistPotential,
                                                 parameters=parameters)

# ============================================================================
#    Plummer sphere potential
#
cdef class _PlummerPotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double G, GM
    cdef public double m, b, b2

    def __init__(self, double G, double m, double b):

        # have to specify G in the correct units
        self.G = G

        # parameters
        self.GM = G*m
        self.m = m
        self.b = b
        self.b2 = b*b

    def __reduce__(self):
        args = (self.G, self.m, self.b)
        return (_PlummerPotential, args)

    cdef public inline double _value(self, double *r) nogil:
        return -self.GM / sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + self.b2)

    cdef public inline void _gradient(self, double *r, double *grad) nogil:
        cdef double R2b, fac
        R2b = (r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) + self.b2
        fac = self.GM / sqrt(R2b) / R2b

        grad[0] += fac*r[0]
        grad[1] += fac*r[1]
        grad[2] += fac*r[2]

class PlummerPotential(CPotential, CartesianPotential):
    r"""
    PlummerPotential(m, b, units)

    Plummer potential for a spheroid.

    .. math::

        \Phi(r) = -\frac{G M}{\sqrt{r^2 + b^2}}

    Parameters
    ----------
    m : numeric
       Mass.
    b : numeric
        Core concentration.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, b, units):
        self.units = units
        _G = G.decompose(units).value
        parameters = dict(G=_G, m=m, b=b)
        super(PlummerPotential, self).__init__(_PlummerPotential,
                                               parameters=parameters)

# ============================================================================
#    Jaffe spheroid potential
#
cdef class _JaffePotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double G, GM
    cdef public double m, c

    def __init__(self, double G, double m, double c):

        # have to specify G in the correct units
        self.G = G

        # disk parameters
        self.GM = G*m
        self.m = m
        self.c = c

    def __reduce__(self):
        args = (self.G, self.m, self.c)
        return (_JaffePotential, args)

    cdef public inline double _value(self, double *r) nogil:
        cdef double R
        R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
        return self.GM / self.c * log(R / (R + self.c))

    cdef public inline void _gradient(self, double *r, double *grad) nogil:
        cdef double R, fac
        R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])
        fac = self.GM / ((R + self.c) * R * R)

        grad[0] += fac*r[0]
        grad[1] += fac*r[1]
        grad[2] += fac*r[2]

class JaffePotential(CPotential, CartesianPotential):
    r"""
    JaffePotential(m, c, units)

    Jaffe potential for a spheroid.

    .. math::

        \Phi(r) = \frac{G M}{c} \ln(\frac{r}{r + c})

    Parameters
    ----------
    m : numeric
        Mass.
    c : numeric
        Core concentration.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, c, units):
        self.units = units
        _G = G.decompose(units).value
        parameters = dict(G=_G, m=m, c=c)
        super(JaffePotential, self).__init__(_JaffePotential,
                                             parameters=parameters)


# ============================================================================
#    Miyamoto-Nagai Disk potential from Miyamoto & Nagai 1975
#    http://adsabs.harvard.edu/abs/1975PASJ...27..533M
#
cdef class _MiyamotoNagaiPotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double G, GM
    cdef public double m, a, b, b2

    def __init__(self, double G, double m, double a, double b):

        # cdef double G = 4.499753324353494927e-12 # kpc^3 / Myr^2 / M_sun
        # have to specify G in the correct units
        self.G = G

        # disk parameters
        self.GM = G*m
        self.m = m
        self.a = a
        self.b = b
        self.b2 = b*b

    def __reduce__(self):
        args = (self.G, self.m, self.a, self.b)
        return (_MiyamotoNagaiPotential, args)

    cdef public inline double _value(self, double *r) nogil:
        cdef double zd
        zd = (self.a + sqrt(r[2]*r[2] + self.b2))
        return -self.GM / sqrt(r[0]*r[0] + r[1]*r[1] + zd*zd)

    cdef public inline void _gradient(self, double *r, double *grad) nogil:
        cdef double sqrtz, zd, fac

        sqrtz = sqrt(r[2]*r[2] + self.b2)
        zd = self.a + sqrtz
        fac = self.GM*pow(r[0]*r[0] + r[1]*r[1] + zd*zd, -1.5)

        grad[0] += fac*r[0]
        grad[1] += fac*r[1]
        grad[2] += fac*r[2] * (1. + self.a / sqrtz)

class MiyamotoNagaiPotential(CPotential, CartesianPotential):
    r"""
    MiyamotoNagaiPotential(m, a, b, units)

    Miyamoto-Nagai potential for a flattened mass distribution.

    .. math::

        \Phi(R,z) = -\frac{G M}{\sqrt{R^2 + (a + \sqrt{z^2 + b^2})^2}}

    Parameters
    ----------
    m : numeric
        Mass.
    a : numeric
        Scale length.
    b : numeric
        Scare height.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, a, b, units):
        self.units = units
        _G = G.decompose(units).value
        parameters = dict(G=_G, m=m, a=a, b=b)
        super(MiyamotoNagaiPotential, self).__init__(_MiyamotoNagaiPotential,
                                                     parameters=parameters)

# ============================================================================
#    Spherical NFW potential
#
cdef class _SphericalNFWPotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double G, v_c, r_s
    cdef public double v_h, v_h2, r_s2

    def __init__(self, double G, double v_c, double r_s):
        self.G = G
        self.v_h = v_c / sqrt(log(2.) - 0.5)
        self.v_h2 = self.v_h*self.v_h
        self.r_s = r_s
        self.r_s2 = r_s*r_s

    def __reduce__(self):
        args = (self.G, self.v_c, self.r_s)
        return (_SphericalNFWPotential, args)

    cdef public inline double _value(self, double *r) nogil:
        cdef double u
        u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) / self.r_s
        return -self.v_h2 * log(1 + u) / u

    cdef public inline void _gradient(self, double *r, double *grad) nogil:
        cdef double fac, u

        u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) / self.r_s
        fac = self.v_h2 / (u*u*u) / self.r_s2 * (log(1+u) - u/(1+u))

        grad[0] += fac*r[0]
        grad[1] += fac*r[1]
        grad[2] += fac*r[2]

class SphericalNFWPotential(CPotential, CartesianPotential):
    r"""
    SphericalNFWPotential(v_c, r_s, units)

    Spherical NFW potential. Separate from the triaxial potential below to
    optimize for speed. Much faster than computing the triaxial case.

    .. math::

        \Phi(r) = -\frac{v_h^2}{\sqrt{\ln 2 - \frac{1}{2}}} \frac{\ln(1 + r/r_s)}{r/r_s}

    Parameters
    ----------
    v_c : numeric
        Circular velocity at the scale radius.
    r_s : numeric
        Scale radius.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_c, r_s, units):
        self.units = units
        _G = G.decompose(units).value
        parameters = dict(G=_G, v_c=v_c, r_s=r_s)

        super(SphericalNFWPotential, self).__init__(_SphericalNFWPotential,
                                                    parameters=parameters)

# ============================================================================
#    Lee & Suto (2003) triaxial NFW potential
#    http://adsabs.harvard.edu/abs/2003ApJ...585..151L
#
cdef class _LeeSutoTriaxialNFWPotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double v_h, r_s, a, b, c, e_b2, e_c2, G
    cdef public double v_h2, r_s2, a2, b2, c2, x0
    cdef public double[::1] R

    def __init__(self, double G, double v_c, double r_s, double a, double b, double c,
                 double[::1] R):
        """ Units of everything should be in the system:
                kpc, Myr, radian, M_sun
        """

        self.v_h = v_c/sqrt(log(2.)-0.5)
        self.v_h2 = self.v_h*self.v_h
        self.r_s = r_s
        self.r_s2 = r_s*r_s
        self.a = a
        self.a2 = a*a
        self.b = b
        self.b2 = b*b
        self.c = c
        self.c2 = c*c
        self.G = G

        self.e_b2 = 1-pow(b/a,2)
        self.e_c2 = 1-pow(c/a,2)

        self.R = R

    def __reduce__(self):
        args = (self.G, self.v_h*sqrt(log(2.)-0.5), self.r_s,
                self.a, self.b, self.c, np.asarray(self.R))
        return (_LeeSutoTriaxialNFWPotential, args)

    cdef public inline double _value(self, double *r) nogil:
        cdef double x, y, z, _r, u

        x = self.R[0]*r[0] + self.R[1]*r[1] + self.R[2]*r[2]
        y = self.R[3]*r[0] + self.R[4]*r[1] + self.R[5]*r[2]
        z = self.R[6]*r[0] + self.R[7]*r[1] + self.R[8]*r[2]

        _r = sqrt(x*x + y*y + z*z)
        u = _r / self.r_s
        return self.v_h2*((self.e_b2/2 + self.e_c2/2)*((1/u - 1/u**3)*log(u + 1) - 1 + (2*u**2 - 3*u + 6)/(6*u**2)) + (self.e_b2*y**2/(2*_r*_r) + self.e_c2*z*z/(2*_r*_r))*((u*u - 3*u - 6)/(2*u*u*(u + 1)) + 3*log(u + 1)/u/u/u) - log(u + 1)/u)

    cdef public inline void _gradient(self, double *r, double *grad) nogil:
        cdef:
            double x, y, z, _r, _r2, _r4, ax, ay, az
            double x0, x2, x22

            double x20, x21, x7, x1
            double x10, x13, x15, x16, x17

        x = self.R[0]*r[0] + self.R[1]*r[1] + self.R[2]*r[2]
        y = self.R[3]*r[0] + self.R[4]*r[1] + self.R[5]*r[2]
        z = self.R[6]*r[0] + self.R[7]*r[1] + self.R[8]*r[2]

        _r2 = x*x + y*y + z*z
        _r = sqrt(_r2)
        _r4 = _r2*_r2

        x0 = _r + self.r_s
        x1 = x0*x0
        x2 = self.v_h2/(12.*_r4*_r2*_r*x1)
        x10 = log(x0/self.r_s)

        x13 = _r*3.*self.r_s
        x15 = x13 - _r2
        x16 = x15 + 6.*self.r_s2
        x17 = 6.*self.r_s*x0*(_r*x16 - x0*x10*6.*self.r_s2)
        x20 = x0*_r2
        x21 = 2.*_r*x0
        x7 = self.e_b2*y*y + self.e_c2*z*z
        x22 = -12.*_r4*_r*self.r_s*x0 + 12.*_r4*self.r_s*x1*x10 + 3.*self.r_s*x7*(x16*_r2 - 18.*x1*x10*self.r_s2 + x20*(2.*_r - 3.*self.r_s) + x21*(x15 + 9.*self.r_s2)) - x20*(self.e_b2 + self.e_c2)*(-6.*_r*self.r_s*(_r2 - self.r_s2) + 6.*self.r_s*x0*x10*(_r2 - 3.*self.r_s2) + x20*(-4.*_r + 3.*self.r_s) + x21*(-x13 + 2.*_r2 + 6.*self.r_s2))

        ax = x2*x*(x17*x7 + x22)
        ay = x2*y*(x17*(x7 - _r2*self.e_b2) + x22)
        az = x2*z*(x17*(x7 - _r2*self.e_c2) + x22)

        grad[0] += self.R[0]*ax + self.R[3]*ay + self.R[6]*az
        grad[1] += self.R[1]*ax + self.R[4]*ay + self.R[7]*az
        grad[2] += self.R[2]*ax + self.R[5]*ay + self.R[8]*az

class LeeSutoTriaxialNFWPotential(CPotential, CartesianPotential):
    r"""
    LeeSutoTriaxialNFWPotential(v_c, r_s, a, b, c, units, phi=0., theta=0., psi=0.)

    Approximation of a Triaxial NFW Potential with the flattening in the density,
    not the potential. See Lee & Suto (2003) for details.

    Parameters
    ----------
    v_c : numeric
        Circular velocity.
    r_s : numeric
        Scale radius.
    a : numeric
        Major axis.
    b : numeric
        Intermediate axis.
    c : numeric
        Minor axis.
    phi : numeric (optional)
        Euler angle for rotation about z-axis (using the x-convention
        from Goldstein). Allows for specifying a misalignment between
        the halo and disk potentials.
    theta : numeric (optional)
        Euler angle for rotation about x'-axis (using the x-convention
        from Goldstein). Allows for specifying a misalignment between
        the halo and disk potentials.
    psi : numeric (optional)
        Euler angle for rotation about z'-axis (using the x-convention
        from Goldstein). Allows for specifying a misalignment between
        the halo and disk potentials.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_c, r_s, a, b, c, units, phi=0., theta=0., psi=0.):
        self.units = units
        _G = G.decompose(units).value
        parameters = dict(G=_G, v_c=v_c, r_s=r_s, a=a, b=b, c=c)

        if theta != 0 or phi != 0 or psi != 0:
            D = rotation_matrix(phi, "z", unit=u.radian) # TODO: Bad assuming radians
            C = rotation_matrix(theta, "x", unit=u.radian)
            B = rotation_matrix(psi, "z", unit=u.radian)
            R = np.asarray(B.dot(C).dot(D))

        else:
            R = np.eye(3)

        parameters['R'] = np.ravel(R).copy()
        super(LeeSutoTriaxialNFWPotential, self).__init__(_LeeSutoTriaxialNFWPotential,
                                                          parameters=parameters)

# ============================================================================
#    Triaxial, Logarithmic potential
#
cdef class _LogarithmicPotential(_CPotential):

    # here need to cdef all the attributes
    cdef public double v_c, r_h, q1, q2, q3, G
    cdef public double v_c2, r_h2, q1_2, q2_2, q3_2, x0
    cdef public double[::1] R

    def __init__(self, double G, double v_c, double r_h, double q1, double q2, double q3,
                 double[::1] R):

        self.v_c = v_c
        self.v_c2 = v_c*v_c
        self.r_h = r_h
        self.r_h2 = r_h*r_h
        self.q1 = q1
        self.q1_2 = q1*q1
        self.q2 = q2
        self.q2_2 = q2*q2
        self.q3 = q3
        self.q3_2 = q3*q3

        self.R = R
        self.G = G

    def __reduce__(self):
        args = (self.G, self.v_c, self.r_h, self.q1, self.q2, self.q3, np.asarray(self.R))
        return (_LogarithmicPotential, args)

    cdef public inline double _value(self, double *r) nogil:

        cdef double x, y, z

        x = self.R[0]*r[0] + self.R[1]*r[1] + self.R[2]*r[2]
        y = self.R[3]*r[0] + self.R[4]*r[1] + self.R[5]*r[2]
        z = self.R[6]*r[0] + self.R[7]*r[1] + self.R[8]*r[2]

        return 0.5*self.v_c2 * log(x*x/self.q1_2 + y*y/self.q2_2 + z*z/self.q3_2 + self.r_h2)

    cdef public inline void _gradient(self, double *r, double *grad) nogil:

        cdef double x, y, z, _r, _r2, ax, ay, az

        x = self.R[0]*r[0] + self.R[1]*r[1] + self.R[2]*r[2]
        y = self.R[3]*r[0] + self.R[4]*r[1] + self.R[5]*r[2]
        z = self.R[6]*r[0] + self.R[7]*r[1] + self.R[8]*r[2]

        _r2 = x*x + y*y + z*z
        _r = sqrt(_r2)

        fac = self.v_c2/(self.r_h2 + x*x/self.q1_2 + y*y/self.q2_2 + z*z/self.q3_2)
        ax = fac*x/self.q1_2
        ay = fac*y/self.q2_2
        az = fac*z/self.q3_2

        grad[0] += self.R[0]*ax + self.R[3]*ay + self.R[6]*az
        grad[1] += self.R[1]*ax + self.R[4]*ay + self.R[7]*az
        grad[2] += self.R[2]*ax + self.R[5]*ay + self.R[8]*az

class LogarithmicPotential(CPotential, CartesianPotential):
    r"""
    LogarithmicPotential(v_c, r_h, q1, q2, q3, units, phi=0., theta=0., psi=0.)

    Triaxial logarithmic potential.

    .. math::

        \Phi(x,y,z) &= \frac{1}{2}v_{c}^2\ln((x/q_1)^2 + (y/q_2)^2 + (z/q_3)^2 + r_h^2)\\

    Parameters
    ----------
    v_c : numeric
        Circular velocity.
    r_h : numeric
        Scale radius.
    q1 : numeric
        Flattening in X.
    q2 : numeric
        Flattening in Y.
    q3 : numeric
        Flattening in Z.
    phi : numeric (optional)
        Euler angle for rotation about z-axis (using the x-convention
        from Goldstein). Allows for specifying a misalignment between
        the halo and disk potentials.
    theta : numeric (optional)
        Euler angle for rotation about x'-axis (using the x-convention
        from Goldstein). Allows for specifying a misalignment between
        the halo and disk potentials.
    psi : numeric (optional)
        Euler angle for rotation about z'-axis (using the x-convention
        from Goldstein). Allows for specifying a misalignment between
        the halo and disk potentials.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_c, r_h, q1, q2, q3, units, phi=0., theta=0., psi=0.):
        self.units = units
        _G = G.decompose(units).value
        parameters = dict(G=_G, v_c=v_c, r_h=r_h, q1=q1, q2=q2, q3=q3)

        if theta != 0 or phi != 0 or psi != 0:
            D = rotation_matrix(phi, "z", unit=u.radian)  # TODO: Bad assuming radians
            C = rotation_matrix(theta, "x", unit=u.radian)
            B = rotation_matrix(psi, "z", unit=u.radian)
            R = np.asarray(B.dot(C).dot(D))

        else:
            R = np.eye(3)

        parameters['R'] = np.ravel(R).copy()
        super(LogarithmicPotential, self).__init__(_LogarithmicPotential,
                                                   parameters=parameters)
