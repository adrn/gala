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
from ..units import galactic
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
    double henon_heiles_value(double t, double *pars, double *q) nogil
    void henon_heiles_gradient(double t, double *pars, double *q, double *grad) nogil

    double kepler_value(double t, double *pars, double *q) nogil
    void kepler_gradient(double t, double *pars, double *q, double *grad) nogil

    double isochrone_value(double t, double *pars, double *q) nogil
    void isochrone_gradient(double t, double *pars, double *q, double *grad) nogil

    double hernquist_value(double t, double *pars, double *q) nogil
    void hernquist_gradient(double t, double *pars, double *q, double *grad) nogil

    double plummer_value(double t, double *pars, double *q) nogil
    void plummer_gradient(double t, double *pars, double *q, double *grad) nogil

    double jaffe_value(double t, double *pars, double *q) nogil
    void jaffe_gradient(double t, double *pars, double *q, double *grad) nogil

    double stone_value(double t, double *pars, double *q) nogil
    void stone_gradient(double t, double *pars, double *q, double *grad) nogil

    double sphericalnfw_value(double t, double *pars, double *q) nogil
    void sphericalnfw_gradient(double t, double *pars, double *q, double *grad) nogil

    double miyamotonagai_value(double t, double *pars, double *q) nogil
    void miyamotonagai_gradient(double t, double *pars, double *q, double *grad) nogil

    double leesuto_value(double t, double *pars, double *q) nogil
    void leesuto_gradient(double t, double *pars, double *q, double *grad) nogil

    double logarithmic_value(double t, double *pars, double *q) nogil
    void logarithmic_gradient(double t, double *pars, double *q, double *grad) nogil

    double lm10_value(double t, double *pars, double *q) nogil
    void lm10_gradient(double t, double *pars, double *q, double *grad) nogil

    double scf_value(double t, double *pars, double *q) nogil
    void scf_gradient(double t, double *pars, double *q, double *grad) nogil

    double ophiuchus_value(double t, double *pars, double *q) nogil
    void ophiuchus_gradient(double t, double *pars, double *q, double *grad) nogil

__all__ = ['HenonHeilesPotential', 'KeplerPotential', 'HernquistPotential',
           'PlummerPotential', 'MiyamotoNagaiPotential',
           'SphericalNFWPotential', 'LeeSutoTriaxialNFWPotential',
           'LogarithmicPotential', 'JaffePotential',
           'StonePotential', 'IsochronePotential',
           'LM10Potential', 'SCFPotential', 'OphiuchusPotential']

# ============================================================================
#    Hénon-Heiles potential
#
cdef class _HenonHeilesPotential(_CPotential):

    def __cinit__(self):
        self._parvec = np.array([])
        self._parameters = &(self._parvec)[0]
        self.c_value = &henon_heiles_value
        self.c_gradient = &henon_heiles_gradient

class HenonHeilesPotential(CPotentialBase):
    r"""
    HenonHeilesPotential(units=None)

    The Hénon-Heiles potential.

    .. math::

        \Phi(x,y) = \frac{1}{2}(x^2 + y^2 + 2x^2 y - \frac{2}{3}y^3)

    Parameters
    ----------
    units : iterable (optional)
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, units=None):
        self.parameters = dict()
        super(HenonHeilesPotential, self).__init__(units=units)
        if units is None:
            self.G = 1.
        else:
            self.G = G.decompose(units).value
        self.c_instance = _HenonHeilesPotential(G=self.G)

# ============================================================================
#    Kepler potential
#
cdef class _KeplerPotential(_CPotential):

    def __cinit__(self, double G, double m):
        self._parvec = np.array([G,m])
        self._parameters = &(self._parvec)[0]
        self.c_value = &kepler_value
        self.c_gradient = &kepler_gradient

class KeplerPotential(CPotentialBase):
    r"""
    KeplerPotential(m, units)

    The Kepler potential for a point mass.

    .. math::

        \Phi(r) = -\frac{Gm}{r}

    Parameters
    ----------
    m : numeric
        Mass.
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, units):
        self.parameters = dict(m=m)
        super(KeplerPotential, self).__init__(units=units)
        self.G = G.decompose(units).value
        self.c_instance = _KeplerPotential(G=self.G, **self.parameters)

# ============================================================================
#    Isochrone potential
#
cdef class _IsochronePotential(_CPotential):

    def __cinit__(self, double G, double m, double b):
        self._parvec = np.array([G,m,b])
        self._parameters = &(self._parvec)[0]
        self.c_value = &isochrone_value
        self.c_gradient = &isochrone_gradient

class IsochronePotential(CPotentialBase):
    r"""
    IsochronePotential(m, units)

    The Isochrone potential.

    .. math::

        \Phi = -\frac{GM}{\sqrt{r^2+b^2} + b}

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
        self.parameters = dict(m=m, b=b)
        super(IsochronePotential, self).__init__(units=units)
        self.G = G.decompose(units).value
        self.c_instance = _IsochronePotential(G=self.G, **self.parameters)

    def action_angle(self, x, v):
        """
        Transform the input cartesian position and velocity to action-angle
        coordinates the Isochrone potential. See Section 3.5.2 in
        Binney & Tremaine (2008), and be aware of the errata entry for
        Eq. 3.225.

        This transformation is analytic and can be used as a "toy potential"
        in the Sanders & Binney 2014 formalism for computing action-angle
        coordinates in _any_ potential.

        Adapted from Jason Sanders' code
        `here <https://github.com/jlsanders/genfunc>`_.

        Parameters
        ----------
        x : array_like
            Positions.
        v : array_like
            Velocities.
        """
        from ..dynamics.analyticactionangle import isochrone_xv_to_aa
        return isochrone_xv_to_aa(x, v, self)

    def phase_space(self, actions, angles):
        """
        Transform the input actions and angles to ordinary phase space (position
        and velocity) in cartesian coordinates. See Section 3.5.2 in
        Binney & Tremaine (2008), and be aware of the errata entry for
        Eq. 3.225.

        Parameters
        ----------
        actions : array_like
        angles : array_like
        """
        from ..dynamics.analyticactionangle import isochrone_aa_to_xv
        return isochrone_aa_to_xv(actions, angles, self)

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
        self.parameters = dict(m=m, c=c)
        super(HernquistPotential, self).__init__(units=units)
        self.G = G.decompose(units).value
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
        self.parameters = dict(m=m, b=b)
        super(PlummerPotential, self).__init__(units=units)
        self.G = G.decompose(units).value
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
        self.parameters = dict(m=m, c=c)
        super(JaffePotential, self).__init__(units=units)
        self.G = G.decompose(units).value
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
        self.parameters = dict(m=m, a=a, b=b)
        super(MiyamotoNagaiPotential, self).__init__(units=units)
        self.G = G.decompose(units).value
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
        self.parameters = dict(m_tot=m_tot, r_c=r_c, r_t=r_t)
        super(StonePotential, self).__init__(units=units)
        self.G = G.decompose(units).value
        self.c_instance = _StonePotential(G=self.G, **self.parameters)

# ============================================================================
#    Spherical NFW potential
#
cdef class _SphericalNFWPotential(_CPotential):

    def __cinit__(self, double v_c, double r_s):
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
        self.parameters = dict(v_c=v_c, r_s=r_s)
        super(SphericalNFWPotential, self).__init__(units=units)
        self.G = G.decompose(units).value
        self.c_instance = _SphericalNFWPotential(**self.parameters)

# ============================================================================
#    Lee & Suto (2003) triaxial NFW potential
#    http://adsabs.harvard.edu/abs/2003ApJ...585..151L
#
cdef class _LeeSutoTriaxialNFWPotential(_CPotential):

    def __cinit__(self, double v_c, double r_s,
                  double a, double b, double c,
                  double R11, double R12, double R13,
                  double R21, double R22, double R23,
                  double R31, double R32, double R33):
        self._parvec = np.array([v_c,r_s,a,b,c, R11,R12,R13,R21,R22,R23,R31,R32,R33])
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
        self.parameters = dict(v_c=v_c, r_s=r_s, a=a, b=b, c=c)
        super(LeeSutoTriaxialNFWPotential, self).__init__(units=units)
        self.G = G.decompose(units).value

        if R is None:
            if theta != 0 or phi != 0 or psi != 0:
                D = rotation_matrix(phi, "z", unit=u.radian) # TODO: Bad assuming radians
                C = rotation_matrix(theta, "x", unit=u.radian)
                B = rotation_matrix(psi, "z", unit=u.radian)
                R = np.asarray(B.dot(C).dot(D))

            else:
                R = np.eye(3)

        # Note: R is the upper triangle of the rotation matrix
        R = np.ravel(R)
        if R.size != 9:
            raise ValueError("Rotation matrix parameter, R, should have 9 elements.")

        c_params = self.parameters.copy()
        c_params['R11'] = R[0]
        c_params['R12'] = R[1]
        c_params['R13'] = R[2]
        c_params['R21'] = R[3]
        c_params['R22'] = R[4]
        c_params['R23'] = R[5]
        c_params['R31'] = R[6]
        c_params['R32'] = R[7]
        c_params['R33'] = R[8]
        self.c_instance = _LeeSutoTriaxialNFWPotential(**c_params)
        self.parameters['R'] = np.ravel(R).copy()

# ============================================================================
#    Triaxial, Logarithmic potential
#
cdef class _LogarithmicPotential(_CPotential):

    def __cinit__(self, double v_c, double r_h,
                  double q1, double q2, double q3,
                  double R11, double R12, double R13,
                  double R21, double R22, double R23,
                  double R31, double R32, double R33):
        self._parvec = np.array([v_c,r_h,q1,q2,q3, R11,R12,R13,R21,R22,R23,R31,R32,R33])
        self._parameters = &(self._parvec)[0]
        self.c_value = &logarithmic_value
        self.c_gradient = &logarithmic_gradient

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
        self.parameters = dict(v_c=v_c, r_h=r_h, q1=q1, q2=q2, q3=q3)
        super(LogarithmicPotential, self).__init__(units=units)
        self.G = G.decompose(units).value

        if R is None:
            if theta != 0 or phi != 0 or psi != 0:
                D = rotation_matrix(phi, "z", unit=u.radian) # TODO: Bad assuming radians
                C = rotation_matrix(theta, "x", unit=u.radian)
                B = rotation_matrix(psi, "z", unit=u.radian)
                R = np.asarray(B.dot(C).dot(D))

            else:
                R = np.eye(3)

        R = np.ravel(R)
        if R.size != 9:
            raise ValueError("Rotation matrix parameter, R, should have 9 elements.")

        c_params = self.parameters.copy()
        c_params['R11'] = R[0]
        c_params['R12'] = R[1]
        c_params['R13'] = R[2]
        c_params['R21'] = R[3]
        c_params['R22'] = R[4]
        c_params['R23'] = R[5]
        c_params['R31'] = R[6]
        c_params['R32'] = R[7]
        c_params['R33'] = R[8]
        self.c_instance = _LogarithmicPotential(**c_params)
        self.parameters['R'] = np.ravel(R).copy()

# ------------------------------------------------------------------------
# HACK
cdef class _LM10Potential(_CPotential):

    def __cinit__(self, double G, double m_spher, double c,
                  double G2, double m_disk, double a, double b,
                  double v_c, double r_h,
                  double q1, double q2, double q3,
                  double R11, double R12, double R13,
                  double R21, double R22, double R23,
                  double R31, double R32, double R33):
        self._parvec = np.array([G,m_spher,c,
                                 G,m_disk,a,b,
                                 v_c,r_h,q1,q2,q3,
                                 R11,R12,R13,R21,R22,R23,R31,R32,R33])
        self._parameters = &(self._parvec[0])
        self.c_value = &lm10_value
        self.c_gradient = &lm10_gradient

class LM10Potential(CPotentialBase):
    r"""
    LM10Potential(units, bulge=dict(), disk=dict(), halo=dict())

    Three-component Milky Way potential model from Law & Majewski (2010).

    Parameters
    ----------
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    bulge : dict
        Dictionary of parameter values for a :class:`HernquistPotential`.
    disk : dict
        Dictionary of parameter values for a :class:`MiyamotoNagaiPotential`.
    halo : dict
        Dictionary of parameter values for a :class:`LogarithmicPotential`.

    """
    def __init__(self, units=galactic, bulge=dict(), disk=dict(), halo=dict()):
        self.G = G.decompose(units).value
        self.parameters = dict()
        default_bulge = dict(m=3.4E10, c=0.7)
        default_disk = dict(m=1E11, a=6.5, b=0.26)
        default_halo = dict(q1=1.38, q2=1., q3=1.36, r_h=12.,
                            phi=(97*u.degree).decompose(units).value,
                            v_c=np.sqrt(2)*(121.858*u.km/u.s).decompose(units).value,
                            theta=0., psi=0.)

        for k,v in default_disk.items():
            if k not in disk:
                disk[k] = v
        self.parameters['disk'] = disk

        for k,v in default_bulge.items():
            if k not in bulge:
                bulge[k] = v
        self.parameters['bulge'] = bulge

        for k,v in default_halo.items():
            if k not in halo:
                halo[k] = v
        self.parameters['halo'] = halo

        super(LM10Potential, self).__init__(units=units)

        if halo.get('R', None) is None:
            if halo['theta'] != 0 or halo['phi'] != 0 or halo['psi'] != 0:
                D = rotation_matrix(halo['phi'], "z", unit=u.radian) # TODO: Bad assuming radians
                C = rotation_matrix(halo['theta'], "x", unit=u.radian)
                B = rotation_matrix(halo['psi'], "z", unit=u.radian)
                R = np.asarray(B.dot(C).dot(D))

            else:
                R = np.eye(3)
        else:
            R = halo['R']

        R = np.ravel(R)
        if R.size != 9:
            raise ValueError("Rotation matrix parameter, R, should have 9 elements.")

        c_params = dict()

        # bulge
        c_params['G'] = self.G
        c_params['m_spher'] = bulge['m']
        c_params['c'] = bulge['c']

        # disk
        c_params['G2'] = self.G
        c_params['m_disk'] = disk['m']
        c_params['a'] = disk['a']
        c_params['b'] = disk['b']

        # halo
        c_params['v_c'] = halo['v_c']
        c_params['r_h'] = halo['r_h']
        c_params['q1'] = halo['q1']
        c_params['q2'] = halo['q2']
        c_params['q3'] = halo['q3']
        c_params['R11'] = R[0]
        c_params['R12'] = R[1]
        c_params['R13'] = R[2]
        c_params['R21'] = R[3]
        c_params['R22'] = R[4]
        c_params['R23'] = R[5]
        c_params['R31'] = R[6]
        c_params['R32'] = R[7]
        c_params['R33'] = R[8]
        self.c_instance = _LM10Potential(**c_params)

cdef class _SCFPotential(_CPotential):
    # double[:,:,::1] sin_coeff, double[:,:,::1] cos_coeff):
    # np.ndarray[np.float64_t, ndim=3] sin_coeff,
    # np.ndarray[np.float64_t, ndim=3] cos_coeff):
    def __cinit__(self, double G, double m, double r_s,
                  int nmax, int lmax,
                  *args):
        self._parvec = np.concatenate([[G,m,r_s,nmax,lmax],args])
                                       # sin_coeff.ravel(),
                                       # cos_coeff.ravel()])
        self._parameters = &(self._parvec[0])
        self.c_value = &scf_value
        self.c_gradient = &scf_gradient

class SCFPotential(CPotentialBase):
    r"""
    SCFPotential(units, TODO)

    TODO:

    Parameters
    ----------
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    TODO

    """
    def __init__(self, m, r_s, sin_coeff, cos_coeff, units=galactic):
        self.G = G.decompose(units).value
        self.parameters = dict()
        self.parameters['m'] = m
        self.parameters['r_s'] = r_s
        self.parameters['sin_coeff'] = np.array(sin_coeff)
        self.parameters['cos_coeff'] = np.array(cos_coeff)
        super(SCFPotential, self).__init__(units=units)

        nmax = sin_coeff.shape[0]-1
        lmax = sin_coeff.shape[1]-1

        # c_params = self.parameters.copy()
        # c_params['G'] = self.G
        # c_params.pop('sin_coeff')
        # c_params.pop('cos_coeff')
        coeff = np.concatenate((sin_coeff.ravel(), cos_coeff.ravel()))
        params1 = [self.G, self.parameters['m'], self.parameters['r_s'],
                   nmax, lmax]
        c_params = np.array(params1 + coeff.tolist())
        # self.c_instance = _SCFPotential(*coeff, **c_params)
        self.c_instance = _SCFPotential(*c_params)

cdef class _WangZhaoBarPotential(_CPotential):
    def __cinit__(self, double G, double m, double r_s, double alpha, double Omega):
        self._parvec = np.array([G,m,r_s,alpha,Omega])
        self._parameters = &(self._parvec[0])
        self.c_value = &wang_zhao_bar_value
        self.c_gradient = &wang_zhao_bar_gradient

class WangZhaoBarPotential(CPotentialBase):
    r"""
    WangZhaoBarPotential(units, TODO)

    TODO:

    Parameters
    ----------
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    TODO

    """
    def __init__(self, m, r_s, alpha, Omega, units=galactic):
        self.G = G.decompose(units).value
        self.parameters = dict()
        self.parameters['m'] = m
        self.parameters['r_s'] = r_s
        self.parameters['alpha'] = alpha
        self.parameters['Omega'] = Omega
        super(WangZhaoBarPotential, self).__init__(units=units)

        c_params = dict()
        c_params['G'] = self.G
        c_params['m'] = m
        c_params['r_s'] = r_s
        c_params['alpha'] = alpha
        c_params['Omega'] = Omega

        self.c_instance = _WangZhaoBarPotential(**c_params)

cdef class _OphiuchusPotential(_CPotential):

    def __cinit__(self, double G, double m_spher, double c,
                  double G2, double m_disk, double a, double b,
                  double v_c, double r_s,
                  double G3, double m_bar, double a_bar, double alpha, double Omega
                  ):
        # alpha = initial bar angle
        # Omega = pattern speed
        self._parvec = np.array([G,m_spher,c,
                                 G,m_disk,a,b,
                                 v_c, r_s,
                                 G,m_bar,a_bar,alpha,Omega])
        self._parameters = &(self._parvec[0])
        self.c_value = &ophiuchus_value
        self.c_gradient = &ophiuchus_gradient

class OphiuchusPotential(CPotentialBase):
    r"""
    OphiuchusPotential(units, spheroid=dict(), disk=dict(), halo=dict(), bar=dict())

    Four-component Milky Way potential used for modeling the Ophiuchus stream.

    Parameters
    ----------
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    spheroid : dict
        Dictionary of parameter values for a :class:`HernquistPotential`.
    disk : dict
        Dictionary of parameter values for a :class:`MiyamotoNagaiPotential`.
    halo : dict
        Dictionary of parameter values for a :class:`SphericalNFWPotential`.
    bar : dict
        Dictionary of parameter values for a :class:`TODO`.

    """
    def __init__(self, units=galactic, spheroid=dict(), disk=dict(), halo=dict(), bar=dict()):
        self.G = G.decompose(units).value
        self.parameters = dict()
        default_spheroid = dict(m=4E9, c=0.1)
        default_disk = dict(m=5.E10, a=3, b=0.28) # similar to Bovy
        default_halo = dict(v_c=0.21, r_s=30.)
        default_bar = dict(m=1.E10, r_s=2.5, alpha=0.349065850398, Omega=0.06136272990322247) # from Wang, Zhao, et al.

        for k,v in default_disk.items():
            if k not in disk:
                disk[k] = v
        self.parameters['disk'] = disk

        for k,v in default_spheroid.items():
            if k not in spheroid:
                spheroid[k] = v
        self.parameters['spheroid'] = spheroid

        for k,v in default_halo.items():
            if k not in halo:
                halo[k] = v
        self.parameters['halo'] = halo

        for k,v in default_bar.items():
            if k not in bar:
                bar[k] = v
        self.parameters['bar'] = bar

        super(OphiuchusPotential, self).__init__(units=units)

        c_params = dict()

        # bulge
        c_params['G'] = self.G
        c_params['m_spher'] = spheroid['m']
        c_params['c'] = spheroid['c']

        # disk
        c_params['G2'] = self.G
        c_params['m_disk'] = disk['m']
        c_params['a'] = disk['a']
        c_params['b'] = disk['b']

        # halo
        c_params['v_c'] = halo['v_c']
        c_params['r_s'] = halo['r_s']

        # bar
        c_params['G3'] = self.G
        c_params['m_bar'] = bar['m']
        c_params['a_bar'] = bar['r_s']
        c_params['alpha'] = bar['alpha']
        c_params['Omega'] = bar['Omega']

        self.c_instance = _OphiuchusPotential(**c_params)
