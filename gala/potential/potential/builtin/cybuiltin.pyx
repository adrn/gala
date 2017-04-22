# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Built-in potentials implemented in Cython """

from __future__ import division, print_function

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
from ..cpotential cimport CPotential, CPotentialWrapper
from ..cpotential cimport densityfunc, energyfunc, gradientfunc, hessianfunc
from ...frame.cframe cimport CFrameWrapper
from ....units import dimensionless, DimensionlessUnitSystem

cdef extern from "potential/builtin/builtin_potentials.h":
    double henon_heiles_value(double t, double *pars, double *q, int n_dim) nogil
    void henon_heiles_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil

    double kepler_value(double t, double *pars, double *q, int n_dim) nogil
    void kepler_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    void kepler_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double isochrone_value(double t, double *pars, double *q, int n_dim) nogil
    void isochrone_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double isochrone_density(double t, double *pars, double *q, int n_dim) nogil
    void isochrone_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double hernquist_value(double t, double *pars, double *q, int n_dim) nogil
    void hernquist_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double hernquist_density(double t, double *pars, double *q, int n_dim) nogil
    void hernquist_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double plummer_value(double t, double *pars, double *q, int n_dim) nogil
    void plummer_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double plummer_density(double t, double *pars, double *q, int n_dim) nogil
    void plummer_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double jaffe_value(double t, double *pars, double *q, int n_dim) nogil
    void jaffe_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double jaffe_density(double t, double *pars, double *q, int n_dim) nogil

    double stone_value(double t, double *pars, double *q, int n_dim) nogil
    void stone_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double stone_density(double t, double *pars, double *q, int n_dim) nogil

    double sphericalnfw_value(double t, double *pars, double *q, int n_dim) nogil
    void sphericalnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double sphericalnfw_density(double t, double *pars, double *q, int n_dim) nogil
    void sphericalnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double flattenednfw_value(double t, double *pars, double *q, int n_dim) nogil
    void flattenednfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil

    double triaxialnfw_value(double t, double *pars, double *q, int n_dim) nogil
    void triaxialnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil

    double satoh_value(double t, double *pars, double *q, int n_dim) nogil
    void satoh_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double satoh_density(double t, double *pars, double *q, int n_dim) nogil

    double miyamotonagai_value(double t, double *pars, double *q, int n_dim) nogil
    void miyamotonagai_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    void miyamotonagai_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil
    double miyamotonagai_density(double t, double *pars, double *q, int n_dim) nogil

    double leesuto_value(double t, double *pars, double *q, int n_dim) nogil
    void leesuto_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double leesuto_density(double t, double *pars, double *q, int n_dim) nogil

    double logarithmic_value(double t, double *pars, double *q, int n_dim) nogil
    void logarithmic_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil

    double longmuralibar_value(double t, double *pars, double *q, int n_dim) nogil
    void longmuralibar_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil

__all__ = ['HenonHeilesPotential', # Misc. potentials
           'KeplerPotential', 'HernquistPotential', 'IsochronePotential', 'PlummerPotential',
           'JaffePotential', 'StonePotential', # Spherical models
           'SatohPotential', 'MiyamotoNagaiPotential', # Disk models
           'NFWPotential', 'LeeSutoTriaxialNFWPotential', 'LogarithmicPotential',
           'LongMuraliBarPotential', # Triaxial models
           'SphericalNFWPotential', 'FlattenedNFWPotential' # Deprecated
           ]

# ============================================================================

cdef class HenonHeilesWrapper(CPotentialWrapper):

    def __init__(self, G, _, q0):
        self.init([G], np.ascontiguousarray(q0), n_dim=2)
        self.cpotential.value[0] = <energyfunc>(henon_heiles_value)
        self.cpotential.gradient[0] = <gradientfunc>(henon_heiles_gradient)

class HenonHeilesPotential(CPotentialBase):
    r"""
    HenonHeilesPotential(units=None, origin=None)

    The Hénon-Heiles potential.

    .. math::

        \Phi(x,y) = \frac{1}{2}(x^2 + y^2 + 2x^2 y - \frac{2}{3}y^3)

    Parameters
    ----------
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, units=None, origin=None):
        parameters = OrderedDict()
        super(HenonHeilesPotential, self).__init__(parameters=parameters,
                                                   parameter_physical_types=dict(),
                                                   ndim=2,
                                                   units=units,
                                                   origin=origin)


# ============================================================================
# Spherical models
#
cdef class KeplerWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(kepler_value)
        self.cpotential.gradient[0] = <gradientfunc>(kepler_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(kepler_hessian)

class KeplerPotential(CPotentialBase):
    r"""
    KeplerPotential(m, units=None, origin=None)

    The Kepler potential for a point mass.

    .. math::

        \Phi(r) = -\frac{Gm}{r}

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Point mass value.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, units=None, origin=None):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        super(KeplerPotential, self).__init__(parameters=parameters,
                                              parameter_physical_types=ptypes,
                                              units=units,
                                              origin=origin)


cdef class IsochroneWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(isochrone_value)
        self.cpotential.density[0] = <densityfunc>(isochrone_density)
        self.cpotential.gradient[0] = <gradientfunc>(isochrone_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(isochrone_hessian)

class IsochronePotential(CPotentialBase):
    r"""
    IsochronePotential(m, b, units=None, origin=None)

    The Isochrone potential.

    .. math::

        \Phi = -\frac{GM}{\sqrt{r^2+b^2} + b}

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, b, units=None, origin=None):
        ptypes = OrderedDict()
        parameters = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        parameters['b'] = b
        ptypes['b'] = 'length'

        super(IsochronePotential, self).__init__(parameters=parameters,
                                                 parameter_physical_types=ptypes,
                                                 units=units,
                                                 origin=origin)

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
        w : :class:`gala.dynamics.PhaseSpacePosition`, :class:`gala.dynamics.Orbit`
            The positions or orbit to compute the actions, angles, and frequencies at.
        """
        from ....dynamics.analyticactionangle import isochrone_to_aa
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


cdef class HernquistWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(hernquist_value)
        self.cpotential.density[0] = <densityfunc>(hernquist_density)
        self.cpotential.gradient[0] = <gradientfunc>(hernquist_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(hernquist_hessian)

class HernquistPotential(CPotentialBase):
    r"""
    HernquistPotential(m, c, units=None, origin=None)

    Hernquist potential for a spheroid.

    .. math::

        \Phi(r) = -\frac{G M}{r + c}

    See: http://adsabs.harvard.edu/abs/1990ApJ...356..359H

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    c : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, c, units=None, origin=None):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        parameters['c'] = c
        ptypes['c'] = 'length'

        super(HernquistPotential, self).__init__(parameters=parameters,
                                                 parameter_physical_types=ptypes,
                                                 units=units,
                                                 origin=origin)


cdef class PlummerWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(plummer_value)
        self.cpotential.density[0] = <densityfunc>(plummer_density)
        self.cpotential.gradient[0] = <gradientfunc>(plummer_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(plummer_hessian)

class PlummerPotential(CPotentialBase):
    r"""
    PlummerPotential(m, b, units=None, origin=None)

    Plummer potential for a spheroid.

    .. math::

        \Phi(r) = -\frac{G M}{\sqrt{r^2 + b^2}}

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
       Mass.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, b, units=None, origin=None):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        parameters['b'] = b
        ptypes['b'] = 'length'

        super(PlummerPotential, self).__init__(parameters=parameters,
                                               parameter_physical_types=ptypes,
                                               units=units,
                                               origin=origin)


cdef class JaffeWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(jaffe_value)
        self.cpotential.density[0] = <densityfunc>(jaffe_density)
        self.cpotential.gradient[0] = <gradientfunc>(jaffe_gradient)

class JaffePotential(CPotentialBase):
    r"""
    JaffePotential(m, c, units=None, origin=None)

    Jaffe potential for a spheroid.

    .. math::

        \Phi(r) = \frac{G M}{c} \ln(\frac{r}{r + c})

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    c : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, c, units=None, origin=None):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        parameters['c'] = c
        ptypes['c'] = 'length'

        super(JaffePotential, self).__init__(parameters=parameters,
                                             parameter_physical_types=ptypes,
                                             units=units,
                                             origin=origin)


cdef class StoneWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(stone_value)
        self.cpotential.density[0] = <densityfunc>(stone_density)
        self.cpotential.gradient[0] = <gradientfunc>(stone_gradient)

class StonePotential(CPotentialBase):
    r"""
    StonePotential(m, r_c, r_h, units=None, origin=None)

    Stone potential from `Stone & Ostriker (2015) <http://dx.doi.org/10.1088/2041-8205/806/2/L28>`_.

    .. math::

        \Phi(r) = -\frac{2 G M}{\pi(r_h - r_c)}\left[ \frac{\arctan(r/r_h)}{r/r_h} - \frac{\arctan(r/r_c)}{r/r_c} + \frac{1}{2}\ln\left(\frac{r^2+r_h^2}{r^2+r_c^2}\right)\right]

    Parameters
    ----------
    m_tot : :class:`~astropy.units.Quantity`, numeric [mass]
        Total mass.
    r_c : :class:`~astropy.units.Quantity`, numeric [length]
        Core radius.
    r_h : :class:`~astropy.units.Quantity`, numeric [length]
        Halo radius.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, r_c, r_h, units=None, origin=None):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        parameters['r_c'] = r_c
        ptypes['r_c'] = 'length'

        parameters['r_h'] = r_h
        ptypes['r_h'] = 'length'

        super(StonePotential, self).__init__(parameters=parameters,
                                             parameter_physical_types=ptypes,
                                             units=units,
                                             origin=origin)


# ============================================================================
# Flattened, axisymmetric models
#
cdef class SatohWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(satoh_value)
        self.cpotential.density[0] = <densityfunc>(satoh_density)
        self.cpotential.gradient[0] = <gradientfunc>(satoh_gradient)

class SatohPotential(CPotentialBase):
    r"""
    SatohPotential(m, a, b, units=None, origin=None)

    Satoh potential for a flattened mass distribution.

    .. math::

        \Phi(R,z) = -\frac{G M}{\sqrt{R^2 + z^2 + a(a + 2\sqrt{z^2 + b^2})}}

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Scale length.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Scare height.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, a, b, units=None, origin=None):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        parameters['a'] = a
        ptypes['a'] = 'length'

        parameters['b'] = b
        ptypes['b'] = 'length'

        super(SatohPotential, self).__init__(parameters=parameters,
                                             parameter_physical_types=ptypes,
                                             units=units,
                                             origin=origin)


cdef class MiyamotoNagaiWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(miyamotonagai_value)
        self.cpotential.density[0] = <densityfunc>(miyamotonagai_density)
        self.cpotential.gradient[0] = <gradientfunc>(miyamotonagai_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(miyamotonagai_hessian)

class MiyamotoNagaiPotential(CPotentialBase):
    r"""
    MiyamotoNagaiPotential(m, a, b, units=None, origin=None)

    Miyamoto-Nagai potential for a flattened mass distribution.

    .. math::

        \Phi(R,z) = -\frac{G M}{\sqrt{R^2 + (a + \sqrt{z^2 + b^2})^2}}

    See: http://adsabs.harvard.edu/abs/1975PASJ...27..533M

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Scale length.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Scare height.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, a, b, units=None, origin=None):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        parameters['a'] = a
        ptypes['a'] = 'length'

        parameters['b'] = b
        ptypes['b'] = 'length'

        super(MiyamotoNagaiPotential, self).__init__(parameters=parameters,
                                                     parameter_physical_types=ptypes,
                                                     units=units,
                                                     origin=origin)


# ============================================================================
# Triaxial models
#

cdef class SphericalNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(sphericalnfw_value)
        self.cpotential.density[0] = <densityfunc>(sphericalnfw_density)
        self.cpotential.gradient[0] = <gradientfunc>(sphericalnfw_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(sphericalnfw_hessian)

cdef class FlattenedNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(flattenednfw_value)
        self.cpotential.gradient[0] = <gradientfunc>(flattenednfw_gradient)

cdef class TriaxialNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(triaxialnfw_value)
        self.cpotential.gradient[0] = <gradientfunc>(triaxialnfw_gradient)

class NFWPotential(CPotentialBase):
    r"""
    NFWPotential(m, r_s, a=1, b=1, c=1, units=None, origin=None)

    General Navarro-Frenk-White potential. Supports spherical, flattened, and
    triaxiality but the flattening is introduced into the potential, not the
    density, and can therefore lead to unphysical mass distributions. For a
    triaxial NFW potential that supports flattening in the density, see
    :class:`gala.potential.LeeSutoTriaxialNFWPotential`.

    .. math::

        \Phi(r) = -\frac{v_c^2}{\sqrt{\ln 2 - \frac{1}{2}}} \frac{\ln(1 + r/r_s)}{r/r_s}

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Scale mass.
    r_s : :class:`~astropy.units.Quantity`, numeric [length]
        Scale radius.
    a : numeric
        Major axis scale.
    b : numeric
        Intermediate axis scale.
    c : numeric
        Minor axis scale.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m=None, r_s=None, a=1., b=1., c=1., v_c=None, units=None, origin=None):
        # TODO: v_c included in above for backwards-compatibility (and m, r_s default to None)

        if v_c is not None and m is None:
            import warnings
            warnings.warn("NFWPotential now expects a scale mass in the default initializer. "
                          "To initialize from a circular velocity, use the classmethod "
                          "from_circular_velocity() instead instead.", DeprecationWarning)

            parameters = OrderedDict()
            ptypes = OrderedDict()

            parameters['v_c'] = v_c
            ptypes['v_c'] = 'speed'

            parameters['r_s'] = r_s
            ptypes['r_s'] = 'length'

            # get appropriate units:
            parameters = CPotentialBase._prepare_parameters(parameters, ptypes, units)

            # r_ref = r_s for old parametrization
            m = NFWPotential._vc_rs_rref_to_m(parameters['v_c'], parameters['r_s'],
                                              parameters['r_s'])
            m = m.to(units['mass'])

        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        parameters['r_s'] = r_s
        ptypes['r_s'] = 'length'

        if np.allclose([a, b, c], 1.):
            NFWWrapper = SphericalNFWWrapper

        elif np.allclose([a, b], 1.):
            NFWWrapper = FlattenedNFWWrapper
            parameters['c'] = c

        else:
            NFWWrapper = TriaxialNFWWrapper
            parameters['a'] = a
            parameters['b'] = b
            parameters['c'] = c

        super(NFWPotential, self).__init__(parameters=parameters,
                                           parameter_physical_types=ptypes,
                                           units=units,
                                           Wrapper=NFWWrapper)

    @staticmethod
    def from_circular_velocity(v_c, r_s, a=1., b=1., c=1., r_ref=None, units=None, origin=None):
        r"""
        from_circular_velocity(v_c, r_s, a=1., b=1., c=1., r_ref=None, units=None, origin=None)

        Initialize an NFW potential from a circular velocity, scale radius, and
        reference radius for the circular velocity.

        For scale mass :math:`m_s`, scale radius :math:`r_s`, scaled
        reference radius :math:`u_{\rm ref} = r_{\rm ref}/r_s`:

        .. math::

            \frac{G\,m_s}{r_s} = \frac{v_c^2}{u_{\rm ref}} \,
                \left[\frac{u_{\rm ref}}{1+u_{\rm ref}} -
                \frac{\ln(1+u_{\rm ref})}{u_{\rm ref}^2} \right]^{-1}

        Parameters
        ----------
        v_c : :class:`~astropy.units.Quantity`, numeric [velocity]
            Circular velocity at the reference radius ``r_ref`` (see below).
        r_s : :class:`~astropy.units.Quantity`, numeric [length]
            Scale radius.
        a : numeric
            Major axis scale.
        b : numeric
            Intermediate axis scale.
        c : numeric
            Minor axis scale.
        r_ref : :class:`~astropy.units.Quantity`, numeric [length] (optional)
            Reference radius at which the circular velocity is given. By default,
            this is assumed to be the scale radius, ``r_s``.

        """

        if r_ref is None:
            r_ref = r_s

        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['v_c'] = v_c
        ptypes['v_c'] = 'speed'

        parameters['r_s'] = r_s
        ptypes['r_s'] = 'length'

        parameters['r_ref'] = r_ref
        ptypes['r_ref'] = 'length'

        # get appropriate units:
        parameters = CPotentialBase._prepare_parameters(parameters, ptypes, units)

        m = NFWPotential._vc_rs_rref_to_m(parameters['v_c'], parameters['r_s'], parameters['r_ref'])
        m = m.to(units['mass'])

        return NFWPotential(m=m, r_s=r_s, a=a, b=b, c=c, units=units, origin=origin)

    @staticmethod
    def _vc_rs_rref_to_m(v_c, r_s, r_ref):
        uu = r_ref / r_s
        vs2 = v_c**2 / uu / (np.log(1+uu)/uu**2 - 1/(uu*(1+uu)))
        return (r_s*vs2 / G)

# TODO: remove these in next full version
class SphericalNFWPotential(NFWPotential):

    def __init__(self, v_c, r_s, units=None, origin=None):
        import warnings
        warnings.warn("This class is now superseded by the single interface to all NFW "
                      "potentials, `NFWPotential`. Use that instead.", DeprecationWarning)
        super(SphericalNFWPotential, self).__init__(v_c=v_c, r_s=r_s, units=units, origin=origin)

    def save(self, *args, **kwargs):
        raise NotImplementedError("Use NFWPotential instead!")

class FlattenedNFWPotential(NFWPotential):

    def __init__(self, v_c, r_s, q_z, units=None, origin=None):
        import warnings
        warnings.warn("This class is now superseded by the single interface to all NFW "
                      "potentials, `NFWPotential`. Use that instead.", DeprecationWarning)

        super(FlattenedNFWPotential, self).__init__(v_c=v_c, r_s=r_s, c=q_z, units=units,
                                                    origin=origin)

    def save(self, *args, **kwargs):
        raise NotImplementedError("Use NFWPotential instead!")


cdef class LogarithmicWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(logarithmic_value)
        self.cpotential.gradient[0] = <gradientfunc>(logarithmic_gradient)

class LogarithmicPotential(CPotentialBase):
    r"""
    LogarithmicPotential(v_c, r_h, q1, q2, q3, phi=0, theta=0, psi=0, units=None, origin=None)

    Triaxial logarithmic potential.

    .. math::

        \Phi(x,y,z) &= \frac{1}{2}v_{c}^2\ln((x/q_1)^2 + (y/q_2)^2 + (z/q_3)^2 + r_h^2)\\

    Parameters
    ----------
    v_c : :class:`~astropy.units.Quantity`, numeric [velocity]
        Circular velocity.
    r_h : :class:`~astropy.units.Quantity`, numeric [length]
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
    def __init__(self, v_c, r_h, q1, q2, q3, phi=0., units=None, origin=None):
        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['v_c'] = v_c
        ptypes['v_c'] = 'speed'

        parameters['r_h'] = r_h
        ptypes['r_h'] = 'length'

        parameters['q1'] = q1
        parameters['q2'] = q2
        parameters['q3'] = q3

        parameters['phi'] = phi
        ptypes['phi'] = 'angle'

        super(LogarithmicPotential, self).__init__(parameters=parameters,
                                                   parameter_physical_types=ptypes,
                                                   units=units,
                                                   origin=origin)

        if not isinstance(self.units, DimensionlessUnitSystem):
            if self.units['angle'] != u.radian:
                raise ValueError("Angle unit must be radian.")


cdef class LeeSutoTriaxialNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(leesuto_value)
        self.cpotential.density[0] = <densityfunc>(leesuto_density)
        self.cpotential.gradient[0] = <gradientfunc>(leesuto_gradient)

class LeeSutoTriaxialNFWPotential(CPotentialBase):
    r"""
    LeeSutoTriaxialNFWPotential(v_c, r_s, a, b, c, units=None, origin=None)

    Approximation of a Triaxial NFW Potential with the flattening in the density,
    not the potential.
    See `Lee & Suto (2003) <http://adsabs.harvard.edu/abs/2003ApJ...585..151L>`_
    for details.

    Parameters
    ----------
    v_c : `~astropy.units.Quantity`, numeric [velocity]
        Circular velocity at the scale radius.
    r_h : `~astropy.units.Quantity`, numeric [length]
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
    def __init__(self, v_c, r_s, a, b, c, units=None, origin=None):

        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['v_c'] = v_c
        ptypes['v_c'] = 'speed'

        parameters['r_s'] = r_s
        ptypes['r_s'] = 'length'

        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c

        super(LeeSutoTriaxialNFWPotential, self).__init__(parameters=parameters,
                                                          parameter_physical_types=ptypes,
                                                          units=units,
                                                          origin=origin)


cdef class LongMuraliBarWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(longmuralibar_value)
        self.cpotential.gradient[0] = <gradientfunc>(longmuralibar_gradient)

class LongMuraliBarPotential(CPotentialBase):
    r"""
    LongMuraliBarPotential(m, a, b, c, alpha=0, units=None, origin=None)

    A simple, triaxial model for a galaxy bar. This is a softened "needle"
    density distribution with an analytic potential form.
    See `Long & Murali (1992) <http://adsabs.harvard.edu/abs/1992ApJ...397...44L>`_
    for details.

    Parameters
    ----------
    m : `~astropy.units.Quantity`, numeric [mass]
        Mass scale.
    a : `~astropy.units.Quantity`, numeric [length]
        Bar half-length.
    b : `~astropy.units.Quantity`, numeric [length]
        Like the Miyamoto-Nagai ``b`` parameter.
    c : `~astropy.units.Quantity`, numeric [length]
        Like the Miyamoto-Nagai ``c`` parameter.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, a, b, c, alpha=0., units=None, origin=None):

        parameters = OrderedDict()
        ptypes = OrderedDict()

        parameters['m'] = m
        ptypes['m'] = 'mass'

        for name in 'abc':
            parameters[name] = eval(name)
            ptypes[name] = 'length'

        parameters['alpha'] = alpha
        ptypes['alpha'] = 'angle'

        super(LongMuraliBarPotential, self).__init__(parameters=parameters,
                                                     parameter_physical_types=ptypes,
                                                     units=units,
                                                     origin=origin)

