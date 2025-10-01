# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3
# cython: language=c++

""" Built-in potential wrappers """


import warnings
from libc.stdlib cimport malloc, free

from astropy.constants import G
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()


from ..core import CompositePotential, _potential_docstring, PotentialBase
from ..util import format_doc, sympy_wrap
from ..cpotential import CPotentialBase
from ..cpotential cimport CPotential, CPotentialWrapper
from ..cpotential cimport densityfunc, energyfunc, gradientfunc, hessianfunc
from ...common import PotentialParameter
from ...frame.cframe cimport CFrameWrapper
from ....units import dimensionless, DimensionlessUnitSystem
from ...._cconfig cimport USE_GSL

# GSL includes for spline functionality
cdef extern from "gsl/gsl_spline.h":
    ctypedef struct gsl_spline:
        pass
    ctypedef struct gsl_interp_accel:
        pass
    ctypedef struct gsl_interp_type:
        pass

    # GSL interpolation declarations
    cdef extern from "gsl/gsl_interp.h":
        ctypedef struct gsl_interp_type:
            pass
        ctypedef struct gsl_interp:
            pass
        ctypedef struct gsl_interp_accel:
            pass

        const gsl_interp_type *gsl_interp_linear
        const gsl_interp_type *gsl_interp_polynomial
        const gsl_interp_type *gsl_interp_cspline
        const gsl_interp_type *gsl_interp_cspline_periodic
        const gsl_interp_type *gsl_interp_akima
        const gsl_interp_type *gsl_interp_akima_periodic
        const gsl_interp_type *gsl_interp_steffen

        gsl_interp* gsl_interp_alloc(const gsl_interp_type *T, size_t size)
        gsl_interp_accel* gsl_interp_accel_alloc()
        int gsl_interp_init(gsl_interp *interp, const double *xa, const double *ya, size_t size)
        void gsl_interp_free(gsl_interp *interp)
        void gsl_interp_accel_free(gsl_interp_accel *acc)

    cdef extern from "gsl/gsl_spline.h":
        ctypedef struct gsl_spline:
            pass

        gsl_spline* gsl_spline_alloc(const gsl_interp_type *T, size_t size)
        int gsl_spline_init(gsl_spline *spline, const double *xa, const double *ya, size_t size)
        void gsl_spline_free(gsl_spline *spline)

    # GSL spline functions
    gsl_spline *gsl_spline_alloc(const gsl_interp_type *interp_type, size_t size)
    gsl_interp_accel *gsl_interp_accel_alloc()
    int gsl_spline_init(gsl_spline *spline, const double *xa, const double *ya, size_t size)
    void gsl_spline_free(gsl_spline *spline)
    void gsl_interp_accel_free(gsl_interp_accel *accel)

cdef extern from "potential/potential/builtin/builtin_potentials.h":
    ctypedef struct spherical_spline_state:
        gsl_spline *spline
        gsl_interp_accel *acc
        gsl_spline *rho_r_spline
        gsl_spline *rho_r2_spline
        gsl_interp_accel *rho_r_acc
        gsl_interp_accel *rho_r2_acc
        int n_knots
        int method
        double *r_knots
        double *values

    # Spherical spline functions
    double spherical_spline_density_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void spherical_spline_density_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double spherical_spline_density_density(double t, double *pars, double *q, int n_dim, void *state) nogil

    double spherical_spline_mass_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void spherical_spline_mass_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double spherical_spline_mass_density(double t, double *pars, double *q, int n_dim, void *state) nogil

    double spherical_spline_potential_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void spherical_spline_potential_gradient(double t, double *pars, double *q, int n_dim, double *grad, void *state) nogil
    double spherical_spline_potential_density(double t, double *pars, double *q, int n_dim, void *state) nogil


cdef extern from "potential/potential/builtin/multipole.h":
    double mp_potential(double t, double *pars, double *q, int n_dim, void *state) nogil
    void mp_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state) nogil
    double mp_density(double t, double *pars, double *q, int n_dim, void *state) nogil

    double axisym_cylspline_value(double t, double *pars, double *q, int n_dim, void *state) nogil
    void axisym_cylspline_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state) nogil
    double axisym_cylspline_density(double t, double *pars, double *q, int n_dim, void *state) nogil


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


# ============================================================================
# Spherical spline interpolated potentials
#

cdef class SphericalSplineWrapper(CPotentialWrapper):
    """Wrapper for spherical spline interpolated potentials"""

    cdef spherical_spline_state spl_state
    cdef double *r_knots_copy
    cdef double *values_copy
    cdef str spline_value_type

    def __init__(
        self, G, parameters, q0, R, spline_value_type, interpolation_method, n_knots
    ):
        """
        Parameters
        ----------
        spline_value_type : str
            Type of values provided: "density", "mass", or "potential"
        interpolation_method : str
            Interpolation method to use. Names from GSL (e.g., cspline, linear, akima, etc.).
        """
        self.spline_value_type = spline_value_type

        method_to_enum = {
            "linear": 0,
            "polynomial": 1,
            "cspline": 2,
            "cspline_periodic": 3,
            "akima": 4,
            "akima_periodic": 5,
            "steffen": 6,
        }

        self._setup_spline_state(
            parameters,
            method=method_to_enum[interpolation_method],
            n_knots=n_knots
        )

        self.init([G] + list(parameters),
                  np.ascontiguousarray(q0),
                  np.ascontiguousarray(R))

        if USE_GSL == 1:
            # Set the state pointer to our spline state
            self.cpotential.state[0] = <void*>&self.spl_state

            if self.spline_value_type == "density":
                self.cpotential.value[0] = <energyfunc>(spherical_spline_density_value)
                self.cpotential.gradient[0] = <gradientfunc>(spherical_spline_density_gradient)
                self.cpotential.density[0] = <densityfunc>(spherical_spline_density_density)
            elif self.spline_value_type == "mass":
                self.cpotential.value[0] = <energyfunc>(spherical_spline_mass_value)
                self.cpotential.gradient[0] = <gradientfunc>(spherical_spline_mass_gradient)
                self.cpotential.density[0] = <densityfunc>(spherical_spline_mass_density)
            elif self.spline_value_type == "potential":
                self.cpotential.value[0] = <energyfunc>(spherical_spline_potential_value)
                self.cpotential.gradient[0] = <gradientfunc>(spherical_spline_potential_gradient)
                self.cpotential.density[0] = <densityfunc>(spherical_spline_potential_density)
            else:
                raise ValueError(
                    f"Unknown value_type: {self.spline_value_type}. Must be 'density', "
                    "'mass', or 'potential'"
                )

    cdef void _setup_spline_state(self, parameters, method, n_knots):
        """Setup the cached GSL spline state"""
        # Copy parameter arrays to ensure they stay alive
        self.r_knots_copy = <double*>malloc(n_knots * sizeof(double))
        self.values_copy = <double*>malloc(n_knots * sizeof(double))

        # Temporary arrays for density spline setup
        cdef double *rho_r_values = <double*>malloc(n_knots * sizeof(double))
        cdef double *rho_r2_values = <double*>malloc(n_knots * sizeof(double))

        # TODO: double check that the indices are correct here
        cdef int i
        for i in range(n_knots):
            self.r_knots_copy[i] = parameters[i]
            self.values_copy[i] = parameters[i + n_knots]

        # Set up state struct
        self.spl_state.n_knots = n_knots
        self.spl_state.method = method
        self.spl_state.r_knots = self.r_knots_copy
        self.spl_state.values = self.values_copy

        # Select GSL interpolation type
        cdef const gsl_interp_type *interp_type
        if method == 0:
            interp_type = gsl_interp_linear
        elif method == 1:
            interp_type = gsl_interp_polynomial
        elif method == 2:
            interp_type = gsl_interp_cspline
        elif method == 3:
            interp_type = gsl_interp_cspline_periodic
        elif method == 4:
            interp_type = gsl_interp_akima
        elif method == 5:
            interp_type = gsl_interp_akima_periodic
        elif method == 6:
            interp_type = gsl_interp_steffen
        else:
            raise ValueError(f"Unknown interpolation method, index = {method}")

        # Create GSL objects
        self.spl_state.acc = gsl_interp_accel_alloc()
        self.spl_state.spline = gsl_spline_alloc(interp_type, n_knots)
        gsl_spline_init(
            self.spl_state.spline, self.r_knots_copy, self.values_copy, n_knots
        )

        # For density interpolation, need additional splines for efficient integration
        if self.spline_value_type == "density":
            # Create rho(r) * r spline for potential calculations
            for i in range(n_knots):
                rho_r_values[i] = self.values_copy[i] * self.r_knots_copy[i]

            self.spl_state.rho_r_acc = gsl_interp_accel_alloc()
            self.spl_state.rho_r_spline = gsl_spline_alloc(interp_type, n_knots)
            gsl_spline_init(
                self.spl_state.rho_r_spline, self.r_knots_copy, rho_r_values, n_knots
            )

            # Create rho(r) * r**2 spline for gradient calculations
            for i in range(n_knots):
                rho_r2_values[i] = self.values_copy[i] * self.r_knots_copy[i] * self.r_knots_copy[i]

            self.spl_state.rho_r2_acc = gsl_interp_accel_alloc()
            self.spl_state.rho_r2_spline = gsl_spline_alloc(interp_type, n_knots)
            gsl_spline_init(
                self.spl_state.rho_r2_spline, self.r_knots_copy, rho_r2_values, n_knots
            )

        else:
            # For non-density types, set auxiliary splines to NULL
            self.spl_state.rho_r_spline = NULL
            self.spl_state.rho_r2_spline = NULL
            self.spl_state.rho_r_acc = NULL
            self.spl_state.rho_r2_acc = NULL

        # Clean up temporary arrays
        free(rho_r_values)
        free(rho_r2_values)

    def __dealloc__(self):
        """Clean up GSL objects and allocated memory"""
        if USE_GSL == 1:
            if self.spl_state.spline != NULL:
                gsl_spline_free(self.spl_state.spline)
            if self.spl_state.acc != NULL:
                gsl_interp_accel_free(self.spl_state.acc)
            if self.spl_state.rho_r_spline != NULL:
                gsl_spline_free(self.spl_state.rho_r_spline)
            if self.spl_state.rho_r_acc != NULL:
                gsl_interp_accel_free(self.spl_state.rho_r_acc)
            if self.spl_state.rho_r2_spline != NULL:
                gsl_spline_free(self.spl_state.rho_r2_spline)
            if self.spl_state.rho_r2_acc != NULL:
                gsl_interp_accel_free(self.spl_state.rho_r2_acc)

        if self.r_knots_copy != NULL:
            free(self.r_knots_copy)
        if self.values_copy != NULL:
            free(self.values_copy)
