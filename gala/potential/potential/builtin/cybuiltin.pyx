# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

""" Built-in potential wrappers """

# Standard library
import warnings

# Third-party
from astropy.constants import G
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()

# Project
from ..core import CompositePotential, _potential_docstring, PotentialBase
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

    double mn3_value(double t, double *pars, double *q, int n_dim) nogil
    void mn3_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    void mn3_hessian(double t, double *pars, double *q, int n_dim, double *hess) nogil
    double mn3_density(double t, double *pars, double *q, int n_dim) nogil

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

    double burkert_value(double t, double *pars, double *q, int n_dim) nogil
    void burkert_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double burkert_density(double t, double *pars, double *q, int n_dim) nogil


cdef extern from "potential/potential/builtin/multipole.h":
    double mp_potential(double t, double *pars, double *q, int n_dim) nogil
    void mp_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double mp_density(double t, double *pars, double *q, int n_dim) nogil

    double axisym_cylspline_value(double t, double *pars, double *q, int n_dim) nogil
    void axisym_cylspline_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double axisym_cylspline_density(double t, double *pars, double *q, int n_dim) nogil

# cdef extern from "gsl/gsl_interp.h":
#     ctypedef struct gsl_interp_accel:
#         pass

# cdef extern from "gsl/gsl_interp2d.h":
#     ctypedef struct gsl_interp2d_type:
#         pass

#     ctypedef struct gsl_interp2d:
#         pass

#     gsl_interp2d_type * gsl_interp2d_bicubic
#     gsl_interp_accel gsl_interp_accel_alloc()
#     double gsl_interp2d_eval(const gsl_interp2d *, const double xa[], const double ya[], const double za[], const double, const double, gsl_interp_accel *, gsl_interp_accel *)

# cdef extern from "gsl/gsl_spline2d.h":
#     ctypedef struct gsl_spline2d:
#         pass

#     int gsl_spline2d_init(gsl_spline2d *spline, const double xa[], const double ya[], const double za[], size_t xsize, size_t ysize)
#     double gsl_spline2d_eval(const gsl_spline2d *spline, const double x, const double y, gsl_interp_accel *xacc, gsl_interp_accel *yacc)


__all__ = [
    'HenonHeilesWrapper',
    'KeplerWrapper',
    'HernquistWrapper',
    'IsochroneWrapper',
    'PlummerWrapper',
    'JaffeWrapper',
    'StoneWrapper',
    'PowerLawCutoffWrapper',
    'SatohWrapper',
    'KuzminWrapper',
    'MiyamotoNagaiWrapper',
    'MN3ExponentialDiskWrapper',
    'SphericalNFWWrapper',
    'FlattenedNFWWrapper',
    'TriaxialNFWWrapper',
    'LeeSutoTriaxialNFWWrapper',
    'LogarithmicWrapper',
    'LongMuraliBarWrapper',
    'NullWrapper',
    'MultipoleWrapper',
    'CylSplineWrapper'
    'BurkertWrapper'
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


cdef class IsochroneWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(isochrone_value)
        self.cpotential.density[0] = <densityfunc>(isochrone_density)
        self.cpotential.gradient[0] = <gradientfunc>(isochrone_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(isochrone_hessian)


cdef class HernquistWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(hernquist_value)
        self.cpotential.density[0] = <densityfunc>(hernquist_density)
        self.cpotential.gradient[0] = <gradientfunc>(hernquist_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(hernquist_hessian)


cdef class PlummerWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(plummer_value)
        self.cpotential.density[0] = <densityfunc>(plummer_density)
        self.cpotential.gradient[0] = <gradientfunc>(plummer_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(plummer_hessian)


cdef class JaffeWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(jaffe_value)
        self.cpotential.density[0] = <densityfunc>(jaffe_density)
        self.cpotential.gradient[0] = <gradientfunc>(jaffe_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(jaffe_hessian)


cdef class StoneWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(stone_value)
        self.cpotential.density[0] = <densityfunc>(stone_density)
        self.cpotential.gradient[0] = <gradientfunc>(stone_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(stone_hessian)


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

cdef class BurkertWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(burkert_value)
        self.cpotential.density[0] = <densityfunc>(burkert_density)
        self.cpotential.gradient[0] = <gradientfunc>(burkert_gradient)


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


cdef class KuzminWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(kuzmin_value)
        self.cpotential.density[0] = <densityfunc>(kuzmin_density)
        self.cpotential.gradient[0] = <gradientfunc>(kuzmin_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(null_hessian)


cdef class MiyamotoNagaiWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(miyamotonagai_value)
        self.cpotential.density[0] = <densityfunc>(miyamotonagai_density)
        self.cpotential.gradient[0] = <gradientfunc>(miyamotonagai_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(miyamotonagai_hessian)


cdef class MN3ExponentialDiskWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(mn3_value)
        self.cpotential.density[0] = <densityfunc>(mn3_density)
        self.cpotential.gradient[0] = <gradientfunc>(mn3_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(mn3_hessian)


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


cdef class LogarithmicWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(logarithmic_value)
        self.cpotential.gradient[0] = <gradientfunc>(logarithmic_gradient)
        self.cpotential.hessian[0] = <hessianfunc>(logarithmic_hessian)
        self.cpotential.density[0] = <energyfunc>(logarithmic_density)


cdef class LeeSutoTriaxialNFWWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(leesuto_value)
        self.cpotential.density[0] = <densityfunc>(leesuto_density)
        self.cpotential.gradient[0] = <gradientfunc>(leesuto_gradient)


cdef class LongMuraliBarWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))
        self.cpotential.value[0] = <energyfunc>(longmuralibar_value)
        self.cpotential.gradient[0] = <gradientfunc>(longmuralibar_gradient)
        self.cpotential.density[0] = <densityfunc>(longmuralibar_density)
        self.cpotential.hessian[0] = <hessianfunc>(longmuralibar_hessian)


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


# ==============================================================================
# Multipole and flexible potential models
#
cdef class MultipoleWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))

        if USE_GSL == 1:
            self.cpotential.value[0] = <energyfunc>(mp_potential)
            self.cpotential.density[0] = <densityfunc>(mp_density)
            self.cpotential.gradient[0] = <gradientfunc>(mp_gradient)


cdef class CylSplineWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0, R):
        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))

        if USE_GSL == 1:
            self.cpotential.value[0] = <energyfunc>(axisym_cylspline_value)
            self.cpotential.gradient[0] = <gradientfunc>(axisym_cylspline_gradient)
            self.cpotential.density[0] = <densityfunc>(axisym_cylspline_density)
            #self.cpotential.hessian[0] = <hessianfunc>(axisym_cylspline_hessian)
