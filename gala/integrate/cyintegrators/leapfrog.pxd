# cython: language_level=3

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        pass

cdef extern from "nbody_helper.h":
    const int MAX_NBODY

cdef void c_init_velocity(CPotential *p, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil

cdef void c_leapfrog_step(CPotential *p, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil
