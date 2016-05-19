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
from astropy.extern import six
from astropy.utils import InheritDocstrings
from astropy.coordinates.angles import rotation_matrix
from astropy.constants import G
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()

# Project
from ..core import CompositePotential
from ..cpotential import CPotentialBase
from ..cpotential cimport CPotentialWrapper
from ...units import DimensionlessUnitSystem

cdef extern from "src/cpotential.h":
    enum:
        MAX_N_COMPONENTS = 16

    ctypedef double (*densityfunc)(double t, double *pars, double *q) nogil
    ctypedef double (*valuefunc)(double t, double *pars, double *q) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil

    ctypedef struct CPotential:
        int n_components
        int n_dim
        densityfunc density[MAX_N_COMPONENTS]
        valuefunc value[MAX_N_COMPONENTS]
        gradientfunc gradient[MAX_N_COMPONENTS]
        int n_params[MAX_N_COMPONENTS]
        double *parameters[MAX_N_COMPONENTS]

    double c_value(CPotential *p, double t, double *q) nogil
    double c_density(CPotential *p, double t, double *q) nogil
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil

cdef extern from "src/_cbuiltin.h":
    double nan_density(double t, double *pars, double *q) nogil

    double henon_heiles_value(double t, double *pars, double *q) nogil
    void henon_heiles_gradient(double t, double *pars, double *q, double *grad) nogil

    double kepler_value(double t, double *pars, double *q) nogil
    void kepler_gradient(double t, double *pars, double *q, double *grad) nogil

    double isochrone_value(double t, double *pars, double *q) nogil
    void isochrone_gradient(double t, double *pars, double *q, double *grad) nogil
    double isochrone_density(double t, double *pars, double *q) nogil

    double hernquist_value(double t, double *pars, double *q) nogil
    void hernquist_gradient(double t, double *pars, double *q, double *grad) nogil
    double hernquist_density(double t, double *pars, double *q) nogil

    double plummer_value(double t, double *pars, double *q) nogil
    void plummer_gradient(double t, double *pars, double *q, double *grad) nogil
    double plummer_density(double t, double *pars, double *q) nogil

    double jaffe_value(double t, double *pars, double *q) nogil
    void jaffe_gradient(double t, double *pars, double *q, double *grad) nogil
    double jaffe_density(double t, double *pars, double *q) nogil

    double stone_value(double t, double *pars, double *q) nogil
    void stone_gradient(double t, double *pars, double *q, double *grad) nogil
    double stone_density(double t, double *pars, double *q) nogil

    double sphericalnfw_value(double t, double *pars, double *q) nogil
    void sphericalnfw_gradient(double t, double *pars, double *q, double *grad) nogil
    double sphericalnfw_density(double t, double *pars, double *q) nogil

    double flattenednfw_value(double t, double *pars, double *q) nogil
    void flattenednfw_gradient(double t, double *pars, double *q, double *grad) nogil
    double flattenednfw_density(double t, double *pars, double *q) nogil

    double miyamotonagai_value(double t, double *pars, double *q) nogil
    void miyamotonagai_gradient(double t, double *pars, double *q, double *grad) nogil
    double miyamotonagai_density(double t, double *pars, double *q) nogil

    double leesuto_value(double t, double *pars, double *q) nogil
    void leesuto_gradient(double t, double *pars, double *q, double *grad) nogil
    double leesuto_density(double t, double *pars, double *q) nogil

    double logarithmic_value(double t, double *pars, double *q) nogil
    void logarithmic_gradient(double t, double *pars, double *q, double *grad) nogil

    double rotating_logarithmic_value(double t, double *pars, double *q) nogil
    void rotating_logarithmic_gradient(double t, double *pars, double *q, double *grad) nogil

__all__ = ['HenonHeilesPotential', # Misc. potentials
           'KeplerPotential', 'HernquistPotential', 'IsochronePotential', 'PlummerPotential',
           'JaffePotential', 'SphericalNFWPotential', 'StonePotential', # Spherical models
           'MiyamotoNagaiPotential', 'FlattenedNFWPotential', # Flattened models
           'LeeSutoTriaxialNFWPotential', 'LogarithmicPotential', # Triaxial models
           'CCompositePotential']

# ============================================================================

cdef class HenonHeilesWrapper(CPotentialWrapper):

    def __init__(self, G, *args):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(henon_heiles_value)
        cp.density[0] = <densityfunc>(nan_density)
        cp.gradient[0] = <gradientfunc>(henon_heiles_gradient)
        self._params = np.array([G], dtype=np.float64)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 2
        self.cpotential = cp

class HenonHeilesPotential(CPotentialBase):
    r"""
    HenonHeilesPotential(units=None)

    The Hénon-Heiles potential.

    .. math::

        \Phi(x,y) = \frac{1}{2}(x^2 + y^2 + 2x^2 y - \frac{2}{3}y^3)

    Parameters
    ----------
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, units=None):
        parameters = OrderedDict()
        super(HenonHeilesPotential, self).__init__(parameters=parameters,
                                                   units=units)

# ============================================================================

cdef class KeplerWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(kepler_value)
        cp.density[0] = <densityfunc>(nan_density)
        cp.gradient[0] = <gradientfunc>(kepler_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

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

    Parameters
    ----------
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, units):
        parameters = OrderedDict()
        parameters['m'] = m
        super(KeplerPotential, self).__init__(parameters=parameters,
                                              units=units)

# ============================================================================

cdef class IsochroneWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(isochrone_value)
        cp.density[0] = <densityfunc>(isochrone_density)
        cp.gradient[0] = <gradientfunc>(isochrone_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

class IsochronePotential(CPotentialBase):
    r"""
    IsochronePotential(m, b, units)

    The Isochrone potential.

    .. math::

        \Phi = -\frac{GM}{\sqrt{r^2+b^2} + b}

    Parameters
    ----------
    m : numeric
        Mass.
    b : numeric
        Core concentration.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, b, units):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['b'] = b
        super(IsochronePotential, self).__init__(parameters=parameters,
                                                 units=units)

    def action_angle(self, w):
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
        w : :class:`gala.dynamics.CartesianPhaseSpacePosition`, :class:`gala.dynamics.CartesianOrbit`
            The positions or orbit to compute the actions, angles, and frequencies at.
        """
        from ...dynamics.analyticactionangle import isochrone_to_aa
        return isochrone_to_aa(w, self)

    # def phase_space(self, actions, angles):
    #     """
    #     Transform the input actions and angles to ordinary phase space (position
    #     and velocity) in cartesian coordinates. See Section 3.5.2 in
    #     Binney & Tremaine (2008), and be aware of the errata entry for
    #     Eq. 3.225.

    #     Parameters
    #     ----------
    #     actions : array_like
    #     angles : array_like
    #     """
    #     from ...dynamics.analyticactionangle import isochrone_aa_to_xv
    #     return isochrone_aa_to_xv(actions, angles, self)

# ============================================================================

cdef class HernquistWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(hernquist_value)
        cp.density[0] = <densityfunc>(hernquist_density)
        cp.gradient[0] = <gradientfunc>(hernquist_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

class HernquistPotential(CPotentialBase):
    r"""
    HernquistPotential(m, c, units)

    Hernquist potential for a spheroid.

    .. math::

        \Phi(r) = -\frac{G M}{r + c}

    See: http://adsabs.harvard.edu/abs/1990ApJ...356..359H

    Parameters
    ----------
    m : numeric
        Mass.
    c : numeric
        Core concentration.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, c, units):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['c'] = c
        super(HernquistPotential, self).__init__(parameters=parameters,
                                                 units=units)

# ============================================================================

cdef class PlummerWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(plummer_value)
        cp.density[0] = <densityfunc>(plummer_density)
        cp.gradient[0] = <gradientfunc>(plummer_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

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
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, b, units):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['b'] = b
        super(PlummerPotential, self).__init__(parameters=parameters,
                                               units=units)

# ============================================================================

cdef class JaffeWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(jaffe_value)
        cp.density[0] = <densityfunc>(jaffe_density)
        cp.gradient[0] = <gradientfunc>(jaffe_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

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
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, c, units):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['c'] = c
        super(JaffePotential, self).__init__(parameters=parameters,
                                             units=units)

# ============================================================================

cdef class StoneWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(stone_value)
        cp.density[0] = <densityfunc>(stone_density)
        cp.gradient[0] = <gradientfunc>(stone_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

class StonePotential(CPotentialBase):
    r"""
    StonePotential(m, r_c, r_h, units)

    Stone potential from `Stone & Ostriker (2015) <http://dx.doi.org/10.1088/2041-8205/806/2/L28>`_.

    .. math::

        \Phi(r) = -\frac{2 G M}{\pi(r_h - r_c)}\left[ \frac{\arctan(r/r_h)}{r/r_h} - \frac{\arctan(r/r_c)}{r/r_c} + \frac{1}{2}\ln\left(\frac{r^2+r_h^2}{r^2+r_c^2}\right)\right]

    Parameters
    ----------
    m_tot : numeric
        Total mass.
    r_c : numeric
        Core radius.
    r_h : numeric
        Halo radius.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, r_c, r_h, units):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['r_c'] = r_c
        parameters['r_h'] = r_h
        super(StonePotential, self).__init__(parameters=parameters,
                                             units=units)

# ============================================================================

cdef class SphericalNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(sphericalnfw_value)
        cp.density[0] = <densityfunc>(sphericalnfw_density)
        cp.gradient[0] = <gradientfunc>(sphericalnfw_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

class SphericalNFWPotential(CPotentialBase):
    r"""
    SphericalNFWPotential(v_c, r_s, units)

    Spherical NFW potential. Separate from the triaxial potential below to
    optimize for speed. Much faster than computing the triaxial case.

    .. math::

        \Phi(r) = -\frac{v_c^2}{\sqrt{\ln 2 - \frac{1}{2}}} \frac{\ln(1 + r/r_s)}{r/r_s}

    Parameters
    ----------
    v_c : numeric
        Circular velocity at the scale radius.
    r_s : numeric
        Scale radius.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_c, r_s, units):
        parameters = OrderedDict()
        parameters['v_c'] = v_c
        parameters['r_s'] = r_s
        super(SphericalNFWPotential, self).__init__(parameters=parameters,
                                                    units=units)

# ============================================================================

cdef class MiyamotoNagaiWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(miyamotonagai_value)
        cp.density[0] = <densityfunc>(miyamotonagai_density)
        cp.gradient[0] = <gradientfunc>(miyamotonagai_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

class MiyamotoNagaiPotential(CPotentialBase):
    r"""
    MiyamotoNagaiPotential(m, a, b, units)

    Miyamoto-Nagai potential for a flattened mass distribution.

    .. math::

        \Phi(R,z) = -\frac{G M}{\sqrt{R^2 + (a + \sqrt{z^2 + b^2})^2}}

    See: http://adsabs.harvard.edu/abs/1975PASJ...27..533M

    Parameters
    ----------
    m : numeric
        Mass.
    a : numeric
        Scale length.
    b : numeric
        Scare height.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, a, b, units):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['a'] = a
        parameters['b'] = b
        super(MiyamotoNagaiPotential, self).__init__(parameters=parameters,
                                                     units=units)

# ============================================================================

cdef class FlattenedNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(flattenednfw_value)
        cp.density[0] = <densityfunc>(flattenednfw_density)
        cp.gradient[0] = <gradientfunc>(flattenednfw_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

class FlattenedNFWPotential(CPotentialBase):
    r"""
    FlattenedNFWPotential(v_c, r_s, q_z, units)

    Flattened NFW potential. Separate from the triaxial potential below to
    optimize for speed. Much faster than computing the triaxial case.

    .. math::

        \Phi(r) = -\frac{v_c^2}{\sqrt{\ln 2 - \frac{1}{2}}} \frac{\ln(1 + r/r_s)}{r/r_s}\\
        r^2 = x^2 + y^2 + z^2/q_z^2

    Parameters
    ----------
    v_c : numeric
        Circular velocity at the scale radius.
    r_s : numeric
        Scale radius.
    q_z : numeric
        Flattening.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_c, r_s, q_z, units):
        parameters = OrderedDict()
        parameters['v_c'] = v_c
        parameters['r_s'] = r_s
        parameters['q_z'] = q_z
        super(FlattenedNFWPotential, self).__init__(parameters=parameters,
                                                    units=units)

# ============================================================================
#

cdef class LeeSutoTriaxialNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(leesuto_value)
        cp.density[0] = <densityfunc>(leesuto_density)
        cp.gradient[0] = <gradientfunc>(leesuto_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

class LeeSutoTriaxialNFWPotential(CPotentialBase):
    r"""
    LeeSutoTriaxialNFWPotential(v_c, r_s, a, b, c, units)

    Approximation of a Triaxial NFW Potential with the flattening in the density,
    not the potential. See Lee & Suto (2003) for details.

    See: http://adsabs.harvard.edu/abs/2003ApJ...585..151L

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
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_c, r_s, a, b, c, units):
        parameters = OrderedDict()
        parameters['v_c'] = v_c
        parameters['r_s'] = r_s
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c
        super(LeeSutoTriaxialNFWPotential, self).__init__(parameters=parameters,
                                                          units=units)

# ============================================================================

cdef class LogarithmicWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(logarithmic_value)
        cp.density[0] = <densityfunc>(nan_density)
        cp.gradient[0] = <gradientfunc>(logarithmic_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

class LogarithmicPotential(CPotentialBase):
    r"""
    LogarithmicPotential(v_c, r_h, q1, q2, q3, phi=0, theta=0, psi=0, units)

    Triaxial logarithmic potential.

    .. math::

        \Phi(x,y,z) &= \frac{1}{2}v_{c}^2\ln((x/q_1)^2 + (y/q_2)^2 + (z/q_3)^2 + r_h^2)\\

    Parameters
    ----------
    v_c : `~astropy.units.Quantity`, numeric
        Circular velocity.
    r_h : `~astropy.units.Quantity`, numeric
        Scale radius.
    q1 : numeric
        Flattening in X.
    q2 : numeric
        Flattening in Y.
    q3 : numeric
        Flattening in Z.
    phi : `~astropy.units.Quantity`, numeric
        First euler angle in the z-x-z convention.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, v_c, r_h, q1, q2, q3, phi=0., units=None):
        parameters = OrderedDict()
        parameters['v_c'] = v_c
        parameters['r_h'] = r_h
        parameters['q1'] = q1
        parameters['q2'] = q2
        parameters['q3'] = q3
        parameters['phi'] = phi
        super(LogarithmicPotential, self).__init__(parameters=parameters,
                                                   units=units)

        if not isinstance(self.units, DimensionlessUnitSystem):
            if self.units['angle'] != u.radian:
                raise ValueError("Angle unit must be radian.")

# ============================================================================
# TODO: why do these have to be in this file?

cdef class CCompositePotentialWrapper(CPotentialWrapper):

    def __init__(self, list potentials):
        cdef:
            CPotential cp
            CPotential tmp_cp
            int i
            CPotentialWrapper[::1] _cpotential_arr

        _cpotential_arr = np.array(potentials)

        n_components = len(potentials)
        self._n_params = np.zeros(n_components, dtype=np.int32)
        for i in range(n_components):
            self._n_params[i] = _cpotential_arr[i]._n_params[0]

        cp.n_components = n_components
        cp.n_params = &(self._n_params[0])
        cp.n_dim = 0

        for i in range(n_components):
            tmp_cp = _cpotential_arr[i].cpotential
            cp.parameters[i] = &(_cpotential_arr[i]._params[0])
            cp.value[i] = tmp_cp.value[0]
            cp.density[i] = tmp_cp.density[0]
            cp.gradient[i] = tmp_cp.gradient[0]

            if cp.n_dim == 0:
                cp.n_dim = tmp_cp.n_dim
            elif cp.n_dim != tmp_cp.n_dim:
                raise ValueError("Input potentials must have same number of coordinate dimensions")

        self.cpotential = cp

class CCompositePotential(CPotentialBase, CompositePotential):

    def __init__(self, **potentials):
        CompositePotential.__init__(self, **potentials)

    def _reset_c_instance(self):
        self._potential_list = []
        for p in self.values():
            self._potential_list.append(p.c_instance)
        self.G = p.G
        self.c_instance = CCompositePotentialWrapper(self._potential_list)

    def __setitem__(self, *args, **kwargs):
        CompositePotential.__setitem__(self, *args, **kwargs)
        self._reset_c_instance()

    def __reduce__(self):
        """ Properly package the object for pickling """
        derp = tuple([self.units] + [c.parameters for c in self.values()])
        return (self.__class__, derp)
