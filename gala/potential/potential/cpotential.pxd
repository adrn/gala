# cython: language_level=3

# cdef extern from "potential/src/cpotential.h":
#     ctypedef struct CPotential:
#         pass

cdef extern from "src/funcdefs.h":
    ctypedef double (*densityfunc)(double t, double *pars, double *q) nogil
    ctypedef double (*energyfunc)(double t, double *pars, double *q) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, double *hess) nogil

cdef extern from "potential/src/cpotential.h":
    const int MAX_N_COMPONENTS

    ctypedef struct CPotential:
        int n_components
        int n_dim
        int null
        densityfunc density[MAX_N_COMPONENTS]
        energyfunc value[MAX_N_COMPONENTS]
        gradientfunc gradient[MAX_N_COMPONENTS]
        hessianfunc hessian[MAX_N_COMPONENTS]
        int n_params[MAX_N_COMPONENTS]
        double *parameters[MAX_N_COMPONENTS]
        double *q0[MAX_N_COMPONENTS]
        double *R[MAX_N_COMPONENTS]

    double c_potential(CPotential *p, double t, double *q) nogil
    double c_density(CPotential *p, double t, double *q) nogil
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil
    void c_hessian(CPotential *p, double t, double *q, double *hess) nogil

    double c_d_dr(CPotential *p, double t, double *q, double *epsilon) nogil
    double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) nogil
    double c_mass_enclosed(CPotential *p, double t, double *q, double G, double *epsilon) nogil

cpdef _validate_pos_arr(double[:,::1] arr)

cdef class CPotentialWrapper:
    cdef CPotential cpotential
    cdef double[::1] _params
    cdef int[::1] _n_params
    cdef list _potentials # HACK: for CCompositePotentialWrapper
    cdef double[::1] _q0
    cdef double[::1] _R

    cpdef init(self, list parameters, double[::1] q0, double[:, ::1] R,
               int n_dim=?)

    cpdef energy(self, double[:,::1] q, double[::1] t)
    cpdef density(self, double[:,::1] q, double[::1] t)
    cpdef gradient(self, double[:,::1] q, double[::1] t)
    cpdef hessian(self, double[:,::1] q, double[::1] t)

    cpdef d_dr(self, double[:,::1] q, double G, double[::1] t)
    cpdef d2_dr2(self, double[:,::1] q, double G, double[::1] t)
    cpdef mass_enclosed(self, double[:,::1] q, double G, double[::1] t)
