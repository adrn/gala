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
from .cpotential import CPotentialBase

cdef extern from "math.h":
    double sqrt(double x) nogil
    double cbrt(double x) nogil
    double sin(double x) nogil
    double cos(double x) nogil
    double log(double x) nogil
    double fabs(double x) nogil
    double exp(double x) nogil
    double atan(double x) nogil
    double pow(double x, double n) nogil

cdef extern from "_cbuiltin.h":
    double hernquist_value(double *pars, double *q) nogil
    void hernquist_gradient(double *pars, double *q, double *grad) nogil

    double plummer_value(double *pars, double *q) nogil
    void plummer_gradient(double *pars, double *q, double *grad) nogil

    double jaffe_value(double *pars, double *q) nogil
    void jaffe_gradient(double *pars, double *q, double *grad) nogil

    double stone_value(double *pars, double *q) nogil
    void stone_gradient(double *pars, double *q, double *grad) nogil

    double sphericalnfw_value(double *pars, double *q) nogil
    void sphericalnfw_gradient(double *pars, double *q, double *grad) nogil

    double miyamotonagai_value(double *pars, double *q) nogil
    void miyamotonagai_gradient(double *pars, double *q, double *grad) nogil

    double leesuto_value(double *pars, double *q) nogil
    void leesuto_gradient(double *pars, double *q, double *grad) nogil

__all__ = ['HernquistPotential', 'PlummerPotential', 'MiyamotoNagaiPotential',
           'SphericalNFWPotential', 'LeeSutoTriaxialNFWPotential', 'LogarithmicPotential',
           'JaffePotential', 'StonePotential']

# ============================================================================
#    Hernquist Spheroid potential from Hernquist 1990
#    http://adsabs.harvard.edu/abs/1990ApJ...356..359H
#
cdef class _HernquistPotential(_CPotential):

    def __cinit__(self, double G, double m, double c):
        self._parvec = np.array([G,m,c])
        self._parameters = &(self._parvec)[0]
        self.c_value = &hernquist_value
        self.c_gradient = &hernquist_gradient

class HernquistPotential(CPotentialBase):
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
        self.G = G.decompose(units).value
        self.parameters = dict(m=m, c=c)
        self.c_instance = _HernquistPotential(G=self.G, **self.parameters)

# ============================================================================
#    Plummer sphere potential
#
cdef class _PlummerPotential(_CPotential):

    def __cinit__(self, double G, double m, double b):
        self._parvec = np.array([G,m,b])
        self._parameters = &(self._parvec)[0]
        self.c_value = &plummer_value
        self.c_gradient = &plummer_gradient

class PlummerPotential(CPotentialBase):
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
        self.G = G.decompose(units).value
        self.parameters = dict(m=m, b=b)
        self.c_instance = _PlummerPotential(G=self.G, **self.parameters)

# ============================================================================
#    Jaffe spheroid potential
#
cdef class _JaffePotential(_CPotential):

    def __cinit__(self, double G, double m, double c):
        self._parvec = np.array([G,m,c])
        self._parameters = &(self._parvec)[0]
        self.c_value = &jaffe_value
        self.c_gradient = &jaffe_gradient

class JaffePotential(CPotentialBase):
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
        self.G = G.decompose(units).value
        self.parameters = dict(m=m, c=c)
        self.c_instance = _JaffePotential(G=self.G, **self.parameters)


# ============================================================================
#    Miyamoto-Nagai Disk potential from Miyamoto & Nagai 1975
#    http://adsabs.harvard.edu/abs/1975PASJ...27..533M
#
cdef class _MiyamotoNagaiPotential(_CPotential):

    def __cinit__(self, double G, double m, double a, double b):
        self._parvec = np.array([G,m,a,b])
        self._parameters = &(self._parvec)[0]
        self.c_value = &miyamotonagai_value
        self.c_gradient = &miyamotonagai_gradient

class MiyamotoNagaiPotential(CPotentialBase):
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
        self.G = G.decompose(units).value
        self.parameters = dict(m=m, a=a, b=b)
        self.c_instance = _MiyamotoNagaiPotential(G=self.G, **self.parameters)

# ============================================================================
#    Stone and Ostriker potential (2015)
#
cdef class _StonePotential(_CPotential):

    def __cinit__(self, double G, double m_tot, double r_c, double r_t):
        self._parvec = np.array([G,m_tot,r_c,r_t])
        self._parameters = &(self._parvec)[0]
        self.c_value = &stone_value
        self.c_gradient = &stone_gradient

class StonePotential(CPotentialBase):
    r"""
    StonePotential(m_tot, r_c, r_t, units)

    Stone potential from Stone & Ostriker (2015).

    .. math::

        \Phi(r) = -\frac{wrong}{wrong}\left[ \frac{\arctan(r/r_t)}{r/r_t} - \frac{\arctan(r/r_c)}{r/r_c} + \frac{1}{2}\ln\left(\frac{r^2+r_t^2}{r^2+r_c^2}\right)\right]

    Parameters
    ----------
    m_tot : numeric
        Total mass.
    r_c : numeric
        Core radius.
    r_t : numeric
        Truncation radius.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m_tot, r_c, r_t, units):
        self.units = units
        self.G = G.decompose(units).value
        self.parameters = dict(m_tot=m_tot, r_c=r_c, r_t=r_t)
        self.c_instance = _StonePotential(G=self.G, **self.parameters)

# ============================================================================
#    Spherical NFW potential
#
cdef class _SphericalNFWPotential(_CPotential):

    def __cinit__(self, double G, double v_c, double r_s):
        self._parvec = np.array([v_c,r_s])
        self._parameters = &(self._parvec)[0]
        self.c_value = &sphericalnfw_value
        self.c_gradient = &sphericalnfw_gradient

class SphericalNFWPotential(CPotentialBase):
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
        self.G = G.decompose(units).value
        self.parameters = dict(v_c=v_c, r_s=r_s)
        self.c_instance = _SphericalNFWPotential(**self.parameters)

# ============================================================================
#    Lee & Suto (2003) triaxial NFW potential
#    http://adsabs.harvard.edu/abs/2003ApJ...585..151L
#
cdef class _LeeSutoTriaxialNFWPotential(_CPotential):

    def __cinit__(self, double v_c, double r_s,
                  double a, double b, double c,
                  double R11, double R12, double R13, double R22, double R23, double R33):
        self._parvec = np.array([v_c,r_s,a,b,c, R11,R12,R13,R22,R23,R33])
        self._parameters = &(self._parvec)[0]
        self.c_value = &leesuto_value
        self.c_gradient = &leesuto_gradient

class LeeSutoTriaxialNFWPotential(CPotentialBase):
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
    def __init__(self, v_c, r_s, a, b, c, units, phi=0., theta=0., psi=0., R=None):
        self.units = units
        self.G = G.decompose(units).value
        self.parameters = dict(v_c=v_c, r_s=r_s, a=a, b=b, c=c)

        if R is None:
            if theta != 0 or phi != 0 or psi != 0:
                D = rotation_matrix(phi, "z", unit=u.radian) # TODO: Bad assuming radians
                C = rotation_matrix(theta, "x", unit=u.radian)
                B = rotation_matrix(psi, "z", unit=u.radian)
                R = np.asarray(B.dot(C).dot(D))
                R = np.array([R[0,0],R[0,1],R[0,2],R[1,1],R[1,2],R[2,2]])

            else:
                R = np.array([1., 0, 0, 1, 0, 1])

        # Note: R is the upper triangle of the rotation matrix
        R = np.ravel(R)
        if R.size != 6:
            raise ValueError("Rotation matrix parameter, R, should specify the upper triangle "
                             "of a rotation matrix.")

        c_params = self.parameters.copy()
        c_params['R11'] = R[0]
        c_params['R12'] = R[1]
        c_params['R13'] = R[2]
        c_params['R22'] = R[3]
        c_params['R23'] = R[4]
        c_params['R33'] = R[5]
        self.c_instance = _LeeSutoTriaxialNFWPotential(**c_params)
        self.parameters['R'] = np.ravel(R).copy()

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

class LogarithmicPotential(CPotentialBase):
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
    def __init__(self, v_c, r_h, q1, q2, q3, units, phi=0., theta=0., psi=0., R=None):
        self.units = units
        self.G = G.decompose(units).value
        self.parameters = dict(v_c=v_c, r_h=r_h, q1=q1, q2=q2, q3=q3)

        if R is None:
            if theta != 0 or phi != 0 or psi != 0:
                D = rotation_matrix(phi, "z", unit=u.radian) # TODO: Bad assuming radians
                C = rotation_matrix(theta, "x", unit=u.radian)
                B = rotation_matrix(psi, "z", unit=u.radian)
                R = np.asarray(B.dot(C).dot(D))

            else:
                R = np.eye(3)

        self.parameters['R'] = np.ravel(R).copy()
        self.c_instance = _LogarithmicPotential(G=self.G, **self.parameters)
