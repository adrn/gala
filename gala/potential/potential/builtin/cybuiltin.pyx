# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

""" Built-in potentials implemented in Cython """

# Standard library
from collections import OrderedDict
import warnings

# Third-party
from astropy.constants import G
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()

# Project
from ..core import CompositePotential, _potential_docstring
from ..util import format_doc
from ..cpotential import CPotentialBase
from ..cpotential cimport CPotential, CPotentialWrapper
from ..cpotential cimport densityfunc, energyfunc, gradientfunc, hessianfunc
from ...frame.cframe cimport CFrameWrapper
from ....units import dimensionless, DimensionlessUnitSystem

cdef extern from "extra_compile_macros.h":
    int USE_GSL

cdef extern from "potential/potential/builtin/builtin_potentials.h":
    double null_value(double t, double *pars, double *q, int n_dim) nogil
    void null_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double null_density(double t, double *pars, double *q, int n_dim) nogil
    void null_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double henon_heiles_value(double t, double *pars, double *q, int n_dim) nogil
    void henon_heiles_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil

    double kepler_value(double t, double *pars, double *q, int n_dim) nogil
    void kepler_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double kepler_density(double t, double *pars, double *q, int n_dim) nogil
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

    double powerlawcutoff_value(double t, double *pars, double *q, int n_dim) nogil
    void powerlawcutoff_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double powerlawcutoff_density(double t, double *pars, double *q, int n_dim) nogil

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
    double longmuralibar_density(double t, double *pars, double *q, int n_dim) nogil

__all__ = ['NullPotential', 'HenonHeilesPotential', # Misc. potentials
           'KeplerPotential', 'HernquistPotential', 'IsochronePotential', 'PlummerPotential',
           'JaffePotential', 'StonePotential', 'PowerLawCutoffPotential', # Spherical models
           'SatohPotential', 'MiyamotoNagaiPotential', # Disk models
           'NFWPotential', 'LeeSutoTriaxialNFWPotential', 'LogarithmicPotential',
           'LongMuraliBarPotential', # Triaxial models
           ]

# ============================================================================

cdef class HenonHeilesWrapper(CPotentialWrapper):

    def __init__(self, G, _, q0, R):
        self.init([G],
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R),
                  n_dim=2)
        self.cpotential.value[0] = <energyfunc>(henon_heiles_value)
        self.cpotential.gradient[0] = <gradientfunc>(henon_heiles_gradient)

@format_doc(common_doc=_potential_docstring)
class HenonHeilesPotential(CPotentialBase):
    r"""
    HenonHeilesPotential(units=None, origin=None, R=None)

    The Hénon-Heiles potential.

    Parameters
    ----------
    {common_doc}
    """
    def __init__(self, units=None, origin=None, R=None):
        parameters = OrderedDict()
        super(HenonHeilesPotential, self).__init__(parameters=parameters,
                                                   ndim=2,
                                                   units=units,
                                                   origin=origin,
                                                   R=R)

    def to_latex(self):
        return r"\Phi(x,y) = \frac{1}{2}(x^2 + y^2 + 2x^2 y - \frac{2}{3}y^3)"


# ============================================================================
# Spherical models
#
cdef class KeplerWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(kepler_value)
        self.cpotential.density[0] = <densityfunc>(kepler_density)
        self.cpotential.gradient[0] = <gradientfunc>(kepler_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(kepler_hessian)

@format_doc(common_doc=_potential_docstring)
class KeplerPotential(CPotentialBase):
    r"""
    KeplerPotential(m, units=None, origin=None, R=None)

    The Kepler potential for a point mass.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Point mass value.
    {common_doc}
    """
    _physical_types = {'m': 'mass'}

    def __init__(self, m, units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        super(KeplerPotential, self).__init__(parameters=parameters,
                                              units=units,
                                              origin=origin,
                                              R=R)

    def to_latex(self):
        return r"\Phi(r) = -\frac{Gm}{r}"


cdef class IsochroneWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(isochrone_value)
        self.cpotential.density[0] = <densityfunc>(isochrone_density)
        self.cpotential.gradient[0] = <gradientfunc>(isochrone_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(isochrone_hessian)

@format_doc(common_doc=_potential_docstring)
class IsochronePotential(CPotentialBase):
    r"""
    IsochronePotential(m, b, units=None, origin=None, R=None)

    The Isochrone potential.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'b': 'length'}

    def __init__(self, m, b, units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['b'] = b
        super(IsochronePotential, self).__init__(parameters=parameters,
                                                 units=units,
                                                 origin=origin,
                                                 R=R)

    def to_latex(self):
        return r"\Phi = -\frac{GM}{\sqrt{r^2+b^2} + b}"

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
            The positions or orbit to compute the actions, angles, and
            frequencies at.
        """
        from ....dynamics.analyticactionangle import isochrone_to_aa
        return isochrone_to_aa(w, self)

    # def phase_space(self, actions, angles):
    #     """
    #     Transform the input actions and angles to ordinary phase space
    #     (position and velocity) in cartesian coordinates. See Section 3.5.2 in
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

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(hernquist_value)
        self.cpotential.density[0] = <densityfunc>(hernquist_density)
        self.cpotential.gradient[0] = <gradientfunc>(hernquist_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(hernquist_hessian)

@format_doc(common_doc=_potential_docstring)
class HernquistPotential(CPotentialBase):
    r"""
    HernquistPotential(m, c, units=None, origin=None, R=None)

    Hernquist potential for a spheroid.
    See: http://adsabs.harvard.edu/abs/1990ApJ...356..359H

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    c : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'c': 'length'}
    def __init__(self, m, c, units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['c'] = c
        super(HernquistPotential, self).__init__(
            parameters=parameters, units=units, origin=origin, R=R)

    def to_latex(self):
        return r"\Phi(r) = -\frac{G M}{r + c}"

cdef class PlummerWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(plummer_value)
        self.cpotential.density[0] = <densityfunc>(plummer_density)
        self.cpotential.gradient[0] = <gradientfunc>(plummer_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(plummer_hessian)

@format_doc(common_doc=_potential_docstring)
class PlummerPotential(CPotentialBase):
    r"""
    PlummerPotential(m, b, units=None, origin=None, R=None)

    Plummer potential for a spheroid.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
       Mass.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'b': 'length'}
    def __init__(self, m, b, units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['b'] = b
        super(PlummerPotential, self).__init__(parameters=parameters,
                                               units=units,
                                               origin=origin,
                                               R=R)

    def to_latex(self):
        return r"\Phi(r) = -\frac{G M}{\sqrt{r^2 + b^2}}"


cdef class JaffeWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(jaffe_value)
        self.cpotential.density[0] = <densityfunc>(jaffe_density)
        self.cpotential.gradient[0] = <gradientfunc>(jaffe_gradient)

@format_doc(common_doc=_potential_docstring)
class JaffePotential(CPotentialBase):
    r"""
    JaffePotential(m, c, units=None, origin=None, R=None)

    Jaffe potential for a spheroid.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    c : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'c': 'length'}

    def __init__(self, m, c, units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['c'] = c
        super(JaffePotential, self).__init__(parameters=parameters,
                                             units=units,
                                             origin=origin,
                                             R=R)

    def to_latex(self):
        return r"\Phi(r) = \frac{G M}{c} \ln(\frac{r}{r + c})"


cdef class StoneWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(stone_value)
        self.cpotential.density[0] = <densityfunc>(stone_density)
        self.cpotential.gradient[0] = <gradientfunc>(stone_gradient)

@format_doc(common_doc=_potential_docstring)
class StonePotential(CPotentialBase):
    r"""
    StonePotential(m, r_c, r_h, units=None, origin=None, R=None)

    Stone potential from `Stone & Ostriker (2015)
    <http://dx.doi.org/10.1088/2041-8205/806/2/L28>`_.

    Parameters
    ----------
    m_tot : :class:`~astropy.units.Quantity`, numeric [mass]
        Total mass.
    r_c : :class:`~astropy.units.Quantity`, numeric [length]
        Core radius.
    r_h : :class:`~astropy.units.Quantity`, numeric [length]
        Halo radius.
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'r_c': 'length',
                       'r_h': 'length'}
    def __init__(self, m, r_c, r_h, units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['r_c'] = r_c
        parameters['r_h'] = r_h
        super(StonePotential, self).__init__(parameters=parameters,
                                             units=units,
                                             origin=origin,
                                             R=R)

    def to_latex(self):
        return r"\Phi(r) = -\frac{2 G M}{\pi(r_h - r_c)}\left[ \frac{\arctan(r/r_h)}{r/r_h} - \frac{\arctan(r/r_c)}{r/r_c} + \frac{1}{2}\ln\left(\frac{r^2+r_h^2}{r^2+r_c^2}\right)\right]"


cdef class PowerLawCutoffWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))

        if USE_GSL == 1:
            self.cpotential.value[0] = <energyfunc>(powerlawcutoff_value)
            self.cpotential.density[0] = <densityfunc>(powerlawcutoff_density)
            self.cpotential.gradient[0] = <gradientfunc>(powerlawcutoff_gradient)

@format_doc(common_doc=_potential_docstring)
class PowerLawCutoffPotential(CPotentialBase):
    r"""
    PowerLawCutoffPotential(m, alpha, r_c, units=None, origin=None, R=None)

    A spherical power-law density profile with an exponential cutoff.

    The power law index must be ``0 <= alpha < 3``.

    .. note::

        This potential requires GSL to be installed, and Gala must have been
        built and installed with GSL support enaled (the default behavior).
        See http://gala.adrian.pw/en/latest/install.html for more information.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Total mass.
    alpha : numeric
        Power law index. Must satisfy: ``alpha < 3``
    r_c : :class:`~astropy.units.Quantity`, numeric [length]
        Cutoff radius.
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'alpha': 'dimensionless',
                       'r_c': 'length'}

    def __init__(self, m, alpha, r_c, units=None, origin=None, R=None):
        from ...._cconfig import GSL_ENABLED
        if not GSL_ENABLED:
            raise ValueError("Gala was compiled without GSL and so this "
                             "potential -- PowerLawCutoffPotential -- will not "
                             "work.  See the gala documentation for more "
                             "information about installing and using GSL with "
                             "gala: http://gala.adrian.pw/en/latest/install.html")

        parameters = OrderedDict()
        parameters['m'] = m
        parameters['alpha'] = alpha
        parameters['r_c'] = r_c
        super(PowerLawCutoffPotential, self).__init__(
            parameters=parameters, units=units, origin=origin, R=R)

    def to_latex(self):
        return (r"\rho(r) = \frac{A}{r^\alpha} \, \exp{-\frac{r^2}{c^2}}\\" +
        r"A = \frac{m}{2\pi} \, \frac{c^{\alpha-3}}{\Gamma(\frac{3-\alpha}{2})}")


# ============================================================================
# Flattened, axisymmetric models
#
cdef class SatohWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(satoh_value)
        self.cpotential.density[0] = <densityfunc>(satoh_density)
        self.cpotential.gradient[0] = <gradientfunc>(satoh_gradient)

@format_doc(common_doc=_potential_docstring)
class SatohPotential(CPotentialBase):
    r"""
    SatohPotential(m, a, b, units=None, origin=None, R=None)

    Satoh potential for a flattened mass distribution.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Scale length.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Scare height.
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'a': 'length',
                       'b': 'length'}

    def __init__(self, m, a, b, units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['a'] = a
        parameters['b'] = b
        super(SatohPotential, self).__init__(parameters=parameters,
                                             units=units,
                                             origin=origin,
                                             R=R)

    def to_latex(self):
        return r"\Phi(R,z) = -\frac{G M}{\sqrt{R^2 + z^2 + a(a + 2\sqrt{z^2 + b^2})}}"


cdef class MiyamotoNagaiWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(miyamotonagai_value)
        self.cpotential.density[0] = <densityfunc>(miyamotonagai_density)
        self.cpotential.gradient[0] = <gradientfunc>(miyamotonagai_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(miyamotonagai_hessian)

@format_doc(common_doc=_potential_docstring)
class MiyamotoNagaiPotential(CPotentialBase):
    r"""
    MiyamotoNagaiPotential(m, a, b, units=None, origin=None, R=None)

    Miyamoto-Nagai potential for a flattened mass distribution.

    See: http://adsabs.harvard.edu/abs/1975PASJ...27..533M

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Scale length.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Scare height.
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'a': 'length',
                       'b': 'length'}

    def __init__(self, m, a, b, units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['a'] = a
        parameters['b'] = b
        super(MiyamotoNagaiPotential, self).__init__(
            parameters=parameters, units=units, origin=origin, R=R)

    def to_latex(self):
        return r"\Phi(R,z) = -\frac{G M}{\sqrt{R^2 + (a + \sqrt{z^2 + b^2})^2}}"


# ============================================================================
# Triaxial models
#

cdef class SphericalNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(sphericalnfw_value)
        self.cpotential.density[0] = <densityfunc>(sphericalnfw_density)
        self.cpotential.gradient[0] = <gradientfunc>(sphericalnfw_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(sphericalnfw_hessian)

cdef class FlattenedNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(flattenednfw_value)
        self.cpotential.gradient[0] = <gradientfunc>(flattenednfw_gradient)

cdef class TriaxialNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(triaxialnfw_value)
        self.cpotential.gradient[0] = <gradientfunc>(triaxialnfw_gradient)

@format_doc(common_doc=_potential_docstring)
class NFWPotential(CPotentialBase):
    r"""
    NFWPotential(m, r_s, a=1, b=1, c=1, units=None, origin=None, R=None)

    General Navarro-Frenk-White potential. Supports spherical, flattened, and
    triaxiality but the flattening is introduced into the potential, not the
    density, and can therefore lead to unphysical mass distributions. For a
    triaxial NFW potential that supports flattening in the density, see
    :class:`gala.potential.LeeSutoTriaxialNFWPotential`.

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
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'r_s': 'length',
                       'a': 'dimensionless',
                       'b': 'dimensionless',
                       'c': 'dimensionless'}

    def __init__(self, m=None, r_s=None, a=1., b=1., c=1., v_c=None, units=None,
                 origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['r_s'] = r_s

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
                                           units=units,
                                           Wrapper=NFWWrapper,
                                           origin=origin,
                                           R=R)

    def to_latex(self):
        return r"\Phi(r) = -\frac{v_c^2}{\sqrt{\ln 2 - \frac{1}{2}}} \frac{\ln(1 + r/r_s)}{r/r_s}"

    @staticmethod
    def from_circular_velocity(v_c, r_s, a=1., b=1., c=1., r_ref=None,
                               units=None, origin=None, R=None):
        r"""
        from_circular_velocity(v_c, r_s, a=1., b=1., c=1., r_ref=None, units=None, origin=None, R=None)

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

        if not hasattr(v_c, 'unit'):
            v_c = v_c * units['length'] / units['time']

        if not hasattr(r_s, 'unit'):
            r_s = r_s * units['length']

        if r_ref is None:
            r_ref = r_s

        m = NFWPotential._vc_rs_rref_to_m(v_c, r_s, r_ref)
        m = m.to(units['mass'])

        return NFWPotential(m=m, r_s=r_s, a=a, b=b, c=c,
                            units=units, origin=origin, R=R)

    @staticmethod
    def _vc_rs_rref_to_m(v_c, r_s, r_ref):
        uu = r_ref / r_s
        vs2 = v_c**2 / uu / (np.log(1+uu)/uu**2 - 1/(uu*(1+uu)))
        return (r_s*vs2 / G)


cdef class LogarithmicWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(logarithmic_value)
        self.cpotential.gradient[0] = <gradientfunc>(logarithmic_gradient)

@format_doc(common_doc=_potential_docstring)
class LogarithmicPotential(CPotentialBase):
    r"""
    LogarithmicPotential(v_c, r_h, q1, q2, q3, phi=0, theta=0, psi=0, units=None, origin=None, R=None)

    Triaxial logarithmic potential.

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
    {common_doc}
    """
    _physical_types = {'v_c': 'speed',
                       'r_h': 'length',
                       'q1': 'dimensionless',
                       'q2': 'dimensionless',
                       'q3': 'dimensionless',
                       'phi': 'angle'}

    def __init__(self, v_c, r_h, q1, q2, q3, phi=0., units=None,
                 origin=None, R=None):
        parameters = OrderedDict()
        parameters['v_c'] = v_c
        parameters['r_h'] = r_h
        parameters['q1'] = q1
        parameters['q2'] = q2
        parameters['q3'] = q3
        parameters['phi'] = phi

        super(LogarithmicPotential, self).__init__(
            parameters=parameters, units=units, origin=origin, R=R)

        if not isinstance(self.units, DimensionlessUnitSystem):
            if self.units['angle'] != u.radian:
                raise ValueError("Angle unit must be radian.")

    def to_latex(self):
        return r"\Phi(x,y,z) &= \frac{1}{2}v_{c}^2\ln((x/q_1)^2 + (y/q_2)^2 + (z/q_3)^2 + r_h^2)"


cdef class LeeSutoTriaxialNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(leesuto_value)
        self.cpotential.density[0] = <densityfunc>(leesuto_density)
        self.cpotential.gradient[0] = <gradientfunc>(leesuto_gradient)

@format_doc(common_doc=_potential_docstring)
class LeeSutoTriaxialNFWPotential(CPotentialBase):
    r"""
    LeeSutoTriaxialNFWPotential(v_c, r_s, a, b, c, units=None, origin=None, R=None)

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
    {common_doc}
    """
    _physical_types = {'v_c': 'speed',
                       'r_s': 'length',
                       'a': 'dimensionless',
                       'b': 'dimensionless',
                       'c': 'dimensionless'}

    def __init__(self, v_c, r_s, a, b, c, units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['v_c'] = v_c
        parameters['r_s'] = r_s
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c

        super(LeeSutoTriaxialNFWPotential, self).__init__(
            parameters=parameters, units=units, origin=origin, R=R)


cdef class LongMuraliBarWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(longmuralibar_value)
        self.cpotential.gradient[0] = <gradientfunc>(longmuralibar_gradient)
        self.cpotential.density[0] = <densityfunc>(longmuralibar_density)

@format_doc(common_doc=_potential_docstring)
class LongMuraliBarPotential(CPotentialBase):
    r"""
    LongMuraliBarPotential(m, a, b, c, alpha=0, units=None, origin=None, R=None)

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
    {common_doc}
    """
    _physical_types = {'m': 'mass',
                       'a': 'length',
                       'b': 'length',
                       'c': 'length',
                       'alpha': 'angle'}
    def __init__(self, m, a, b, c, alpha=0., units=None, origin=None, R=None):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c
        parameters['alpha'] = alpha
        super(LongMuraliBarPotential, self).__init__(
            parameters=parameters, units=units, origin=origin, R=R)


# ==============================================================================
# Special
#
cdef class NullWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G],
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(null_value)
        self.cpotential.density[0] = <densityfunc>(null_density)
        self.cpotential.gradient[0] = <gradientfunc>(null_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(null_hessian)
        self.cpotential.null = 1

@format_doc(common_doc=_potential_docstring)
class NullPotential(CPotentialBase):
    r"""
    NullPotential(units=None, origin=None, R=None)

    A null potential with 0 mass. Does nothing.

    Parameters
    ----------
    {common_doc}
    """
    def __init__(self, units=None, origin=None, R=None):
        parameters = OrderedDict()
        super(NullPotential, self).__init__(parameters=parameters,
                                            units=units,
                                            origin=origin,
                                            R=R)
