# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

""" Built-in potentials implemented in Cython """

# HACK: This hack brought to you by a bug in cython, and a solution from here:
# https://stackoverflow.com/questions/57138496/class-level-classmethod-can-only-be-called-on-a-method-descriptor-or-instance
try:
    myclassmethod = __builtins__.classmethod
except AttributeError:
    myclassmethod = __builtins__['classmethod']

# Standard library
import warnings

# Third-party
from astropy.constants import G
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()

# Project
from ..core import CompositePotential, _potential_docstring
from ..util import format_doc, sympy_wrap
from ..cpotential import CPotentialBase
from ..cpotential cimport CPotential, CPotentialWrapper
from ..cpotential cimport densityfunc, energyfunc, gradientfunc, hessianfunc
from ...common import PotentialParameter
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
    void henon_heiles_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

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
    void jaffe_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double powerlawcutoff_value(double t, double *pars, double *q, int n_dim) nogil
    void powerlawcutoff_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double powerlawcutoff_density(double t, double *pars, double *q, int n_dim) nogil
    void powerlawcutoff_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double stone_value(double t, double *pars, double *q, int n_dim) nogil
    void stone_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double stone_density(double t, double *pars, double *q, int n_dim) nogil
    void stone_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double sphericalnfw_value(double t, double *pars, double *q, int n_dim) nogil
    void sphericalnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double sphericalnfw_density(double t, double *pars, double *q, int n_dim) nogil
    void sphericalnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double flattenednfw_value(double t, double *pars, double *q, int n_dim) nogil
    void flattenednfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    void flattenednfw_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double triaxialnfw_value(double t, double *pars, double *q, int n_dim) nogil
    void triaxialnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    void triaxialnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double satoh_value(double t, double *pars, double *q, int n_dim) nogil
    void satoh_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double satoh_density(double t, double *pars, double *q, int n_dim) nogil
    void satoh_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

    double kuzmin_value(double t, double *pars, double *q, int n_dim) nogil
    void kuzmin_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double kuzmin_density(double t, double *pars, double *q, int n_dim) nogil

    double miyamotonagai_value(double t, double *pars, double *q, int n_dim) nogil
    void miyamotonagai_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    void miyamotonagai_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil
    double miyamotonagai_density(double t, double *pars, double *q, int n_dim) nogil

    double leesuto_value(double t, double *pars, double *q, int n_dim) nogil
    void leesuto_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double leesuto_density(double t, double *pars, double *q, int n_dim) nogil

    double logarithmic_value(double t, double *pars, double *q, int n_dim) nogil
    void logarithmic_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    void logarithmic_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil
    double logarithmic_density(double t, double *pars, double *q, int n_dim) nogil

    double longmuralibar_value(double t, double *pars, double *q, int n_dim) nogil
    void longmuralibar_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double longmuralibar_density(double t, double *pars, double *q, int n_dim) nogil
    void longmuralibar_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil

__all__ = ['NullPotential', 'HenonHeilesPotential', # Misc. potentials
           'KeplerPotential', 'HernquistPotential', 'IsochronePotential', 'PlummerPotential',
           'JaffePotential', 'StonePotential', 'PowerLawCutoffPotential', # Spherical models
           'SatohPotential', 'KuzminPotential', 'MiyamotoNagaiPotential', # Disk models
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
        self.cpotential.hessian[0] = <hessianfunc>(henon_heiles_hessian)

@format_doc(common_doc=_potential_docstring)
class HenonHeilesPotential(CPotentialBase):
    r"""
    The Hénon-Heiles potential.

    Parameters
    ----------
    {common_doc}
    """
    ndim = 2
    Wrapper = HenonHeilesWrapper

    @myclassmethod
    @sympy_wrap(var='x y')
    def to_sympy(cls, v, p):
        expr = 1./2 * (v['x']**2 + v['y']**2 +
                      2*v['x']**2 * v['y'] - 2./3*v['y']**3)
        return expr, v, p


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
    The Kepler potential for a point mass.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Point mass value.
    {common_doc}
    """
    m = PotentialParameter('m', physical_type='mass')
    Wrapper = KeplerWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        r = sy.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
        expr = - p['G'] * p['m'] / r
        return expr, v, p


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
    The Isochrone potential.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """
    m = PotentialParameter('m', physical_type='mass')
    b = PotentialParameter('b', physical_type='length')

    Wrapper = IsochroneWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        r = sy.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
        expr = - p['G'] * p['m'] / (sy.sqrt(r**2 + p['b']**2) + p['b'])
        return expr, v, p

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
        from gala.dynamics.actionangle import isochrone_xv_to_aa
        return isochrone_xv_to_aa(w, self)


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
    m = PotentialParameter('m', physical_type='mass')
    c = PotentialParameter('c', physical_type='length')

    Wrapper = HernquistWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        r = sy.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
        expr = - p['G'] * p['m'] / (r + p['c'])
        return expr, v, p


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
    Plummer potential for a spheroid.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
       Mass.
    b : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """
    m = PotentialParameter('m', physical_type='mass')
    b = PotentialParameter('b', physical_type='length')

    Wrapper = PlummerWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        r = sy.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
        expr = - p['G'] * p['m'] / sy.sqrt(r**2 + p['b']**2)
        return expr, v, p


cdef class JaffeWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(jaffe_value)
        self.cpotential.density[0] = <densityfunc>(jaffe_density)
        self.cpotential.gradient[0] = <gradientfunc>(jaffe_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(jaffe_hessian)

@format_doc(common_doc=_potential_docstring)
class JaffePotential(CPotentialBase):
    r"""
    Jaffe potential for a spheroid.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    c : :class:`~astropy.units.Quantity`, numeric [length]
        Core concentration.
    {common_doc}
    """
    m = PotentialParameter('m', physical_type='mass')
    c = PotentialParameter('c', physical_type='length')

    Wrapper = JaffeWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        r = sy.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
        expr = p['G'] * p['m'] / p['c'] * sy.log(r / (r + p['c']))
        return expr, v, p


cdef class StoneWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(stone_value)
        self.cpotential.density[0] = <densityfunc>(stone_density)
        self.cpotential.gradient[0] = <gradientfunc>(stone_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(stone_hessian)

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
    m = PotentialParameter('m', physical_type='mass')
    r_c = PotentialParameter('r_c', physical_type='length')
    r_h = PotentialParameter('r_h', physical_type='length')

    Wrapper = StoneWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        r = sy.sqrt(v['x']**2 + v['y']**2 + v['z']**2)
        A = - 2 * p['G'] * p['m'] / (np.pi * (p['r_h'] - p['r_c']))
        expr = A * (p['r_h'] / r * sy.atan(r / p['r_h']) -
                    p['r_c'] / r * sy.atan(r / p['r_c']) +
                    1./2 * sy.log((r**2 + p['r_h']**2) / (r**2 + p['r_c']**2)))
        return expr, v, p


cdef class PowerLawCutoffWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))

        if USE_GSL == 1:
            self.cpotential.value[0] = <energyfunc>(powerlawcutoff_value)
            self.cpotential.density[0] = <densityfunc>(powerlawcutoff_density)
            self.cpotential.gradient[0] = <gradientfunc>(powerlawcutoff_gradient)
            self.cpotential.hessian[0] = <hessianfunc>(powerlawcutoff_hessian)

@format_doc(common_doc=_potential_docstring)
class PowerLawCutoffPotential(CPotentialBase, GSL_only=True):
    r"""
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
    m = PotentialParameter('m', physical_type='mass')
    alpha = PotentialParameter('alpha', physical_type='dimensionless')
    r_c = PotentialParameter('r_c', physical_type='length')

    Wrapper = PowerLawCutoffWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        G = p['G']
        m = p['m']
        alpha = p['alpha']
        r_c = p['r_c']
        r = sy.sqrt(v['x']**2 + v['y']**2 + v['z']**2)

        expr = (G*alpha*m* sy.lowergamma(3./2 - alpha/2, r**2/r_c**2) /
                (2*r* sy.gamma(5./2 - alpha/2)) +
                G*m* sy.lowergamma(1 - alpha/2, r**2/r_c**2) /
                (r_c* sy.gamma(3./2 - alpha/2)) -
                3*G*m* sy.lowergamma(3./2 - alpha/2, r**2/r_c**2) /
                (2*r*sy.gamma(5./2 - alpha/2)))

        return expr, v, p


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
        self.cpotential.hessian[0] = <hessianfunc>(satoh_hessian)

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
    m = PotentialParameter('m', physical_type='mass')
    a = PotentialParameter('a', physical_type='length')
    b = PotentialParameter('b', physical_type='length')

    Wrapper = SatohWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        R = sy.sqrt(v['x']**2 + v['y']**2)
        z = v['z']
        term = R**2 + z**2 + p['a'] * (p['a'] + 2 * sy.sqrt(z**2 + p['b']**2))
        expr = - p['G'] * p['m'] / sy.sqrt(term)
        return expr, v, p


cdef class KuzminWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(kuzmin_value)
        self.cpotential.density[0] = <densityfunc>(kuzmin_density)
        self.cpotential.gradient[0] = <gradientfunc>(kuzmin_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(null_hessian)

@format_doc(common_doc=_potential_docstring)
class KuzminPotential(CPotentialBase):
    r"""
    KuzminPotential(m, a, units=None, origin=None, R=None)

    Kuzmin potential for a flattened mass distribution.

    Parameters
    ----------
    m : :class:`~astropy.units.Quantity`, numeric [mass]
        Mass.
    a : :class:`~astropy.units.Quantity`, numeric [length]
        Flattening parameter.
    {common_doc}
    """
    m = PotentialParameter('m', physical_type='mass')
    a = PotentialParameter('a', physical_type='length')

    Wrapper = KuzminWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        denom = sy.sqrt(v['x']**2 + v['y']**2 + (p['a'] + sy.Abs(v['z']))**2)
        expr = - p['G'] * p['m'] / denom
        return expr, v, p


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
    m = PotentialParameter('m', physical_type='mass')
    a = PotentialParameter('a', physical_type='length')
    b = PotentialParameter('b', physical_type='length')

    Wrapper = MiyamotoNagaiWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        R = sy.sqrt(v['x']**2 + v['y']**2)
        z = v['z']
        term = R**2 + (p['a'] + sy.sqrt(z**2 + p['b']**2))**2
        expr = - p['G'] * p['m'] / sy.sqrt(term)
        return expr, v, p


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
        self.cpotential.hessian[0] = <hessianfunc>(flattenednfw_hessian)

cdef class TriaxialNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(triaxialnfw_value)
        self.cpotential.gradient[0] = <gradientfunc>(triaxialnfw_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(triaxialnfw_hessian)

@format_doc(common_doc=_potential_docstring)
class NFWPotential(CPotentialBase):
    r"""
    NFWPotential(m, r_s, a=1, b=1, c=1, units=None, origin=None, R=None)

    General Navarro-Frenk-White potential. Supports spherical, flattened, and
    triaxiality but the flattening is introduced into the potential, not the
    density, and can therefore lead to unphysical mass distributions. For a
    triaxial NFW potential that supports flattening in the density, see
    :class:`gala.potential.LeeSutoTriaxialNFWPotential`.

    See also the alternate initializers: `NFWPotential.from_circular_velocity`
    and `NFWPotential.from_M200_c`

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
    m = PotentialParameter('m', physical_type='mass')
    r_s = PotentialParameter('r_s', physical_type='length')
    a = PotentialParameter('a', physical_type='dimensionless', default=1.)
    b = PotentialParameter('b', physical_type='dimensionless', default=1.)
    c = PotentialParameter('c', physical_type='dimensionless', default=1.)

    def _setup_potential(self, parameters, origin=None, R=None, units=None):
        super()._setup_potential(parameters, origin=origin, R=R, units=units)

        a = self.parameters['a']
        b = self.parameters['b']
        c = self.parameters['c']

        if np.allclose([a, b, c], 1.):
            self.Wrapper = SphericalNFWWrapper

        elif np.allclose([a, b], 1.):
            self.Wrapper = FlattenedNFWWrapper

        else:
            self.Wrapper = TriaxialNFWWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        uu = sy.sqrt((v['x'] / p['a']) ** 2 +
                     (v['y'] / p['b']) ** 2 +
                     (v['z'] / p['c']) ** 2) / p['r_s']
        v_h2 = p['G'] * p['m'] / p['r_s']
        expr = - v_h2 * sy.log(1 + uu) / uu
        return expr, v, p

    @staticmethod
    def from_M200_c(M200, c, rho_c=None, units=None, origin=None, R=None):
        r"""
        from_M200_c(M200, c, rho_c=None, units=None, origin=None, R=None)

        Initialize an NFW potential from a virial mass, ``M200``, and a
        concentration, ``c``.

        Parameters
        ----------
        M200 : :class:`~astropy.units.Quantity`, numeric [mass]
            Virial mass, or mass at 200 times the critical density, ``rho_c``.
        c : numeric
            NFW halo concentration.
        rho_c : :class:`~astropy.units.Quantity`, numeric [density]
            Critical density at z=0. If not specified, uses the default astropy
            cosmology to obtain this, `~astropy.cosmology.default_cosmology`.
        """
        if rho_c is None:
            from astropy.constants import G
            from astropy.cosmology import default_cosmology
            cosmo = default_cosmology.get()
            rho_c = 3 * cosmo.H(0.)**2 / (8*np.pi*G)

        Rvir = np.cbrt(M200 / (200*rho_c) / (4/3*np.pi))
        r_s = Rvir / c

        A_NFW = np.log(1 + c) - c / (1 + c)
        m = M200 / A_NFW

        return NFWPotential(m=m, r_s=r_s, a=1., b=1., c=1.,
                            units=units, origin=origin, R=R)

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
        self.cpotential.hessian[0] = <hessianfunc>(logarithmic_hessian)
        self.cpotential.density[0] = <energyfunc>(logarithmic_density)

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

    v_c = PotentialParameter('v_c', physical_type='speed')
    r_h = PotentialParameter('r_h', physical_type='length')
    q1 = PotentialParameter('q1', physical_type='dimensionless', default=1.)
    q2 = PotentialParameter('q2', physical_type='dimensionless', default=1.)
    q3 = PotentialParameter('q3', physical_type='dimensionless', default=1.)
    phi = PotentialParameter('phi', physical_type='angle', default=0.)

    Wrapper = LogarithmicWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy
        r2 = ((v['x'] / p['q1']) ** 2 +
              (v['y'] / p['q2']) ** 2 +
              (v['z'] / p['q3']) ** 2)
        expr = 1./2 * p['v_c']**2 * sy.log(p['r_h']**2 + r2)
        return expr, v, p


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
    v_c = PotentialParameter('v_c', physical_type='speed')
    r_s = PotentialParameter('r_s', physical_type='length')
    a = PotentialParameter('a', physical_type='dimensionless', default=1.)
    b = PotentialParameter('b', physical_type='dimensionless', default=1.)
    c = PotentialParameter('c', physical_type='dimensionless', default=1.)

    Wrapper = LeeSutoTriaxialNFWWrapper

    # TODO: implement to_sympy()


cdef class LongMuraliBarWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(longmuralibar_value)
        self.cpotential.gradient[0] = <gradientfunc>(longmuralibar_gradient)
        self.cpotential.density[0] = <densityfunc>(longmuralibar_density)
        self.cpotential.hessian[0] = <hessianfunc>(longmuralibar_hessian)

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
    m = PotentialParameter('m', physical_type='mass')
    a = PotentialParameter('a', physical_type='length')
    b = PotentialParameter('b', physical_type='length')
    c = PotentialParameter('c', physical_type='length')
    alpha = PotentialParameter('alpha', physical_type='angle', default=0)

    Wrapper = LongMuraliBarWrapper

    @myclassmethod
    @sympy_wrap
    def to_sympy(cls, v, p):
        import sympy as sy

        x = v['x'] * sy.cos(p['alpha']) + v['y'] * sy.sin(p['alpha'])
        y = -v['x'] * sy.sin(p['alpha']) + v['y'] * sy.cos(p['alpha'])
        z = v['z']

        Tm = sy.sqrt((p['a'] - x)**2 + y**2 +
                     (p['b'] + sy.sqrt(p['c']**2 + z**2))**2)
        Tp = sy.sqrt((p['a'] + x)**2 + y**2 +
                     (p['b'] + sy.sqrt(p['c']**2 + z**2))**2)

        expr = (p['G'] * p['m'] / (2*p['a']) *
                sy.log((x - p['a'] + Tm) / (x + p['a'] + Tp)))

        return expr, v, p


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
    Wrapper = NullWrapper
