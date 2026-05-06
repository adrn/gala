# cython: language_level=3
# cython: language=c++

from ...potential.potential.cpotential cimport CPotential

cdef void c_init_velocity(CPotential *p, size_t n, int half_ndim, double t, double dt,
                      double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil

cdef void c_leapfrog_step(CPotential *p, size_t n, int half_ndim, double t, double dt,
                            double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil

cdef void c_init_velocity_nbody(
    CPotential *p, int half_ndim, double t, double dt,
    CPotential **pots, double *x_nbody_jm1, int nbody, int nbody_i,
    double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad
) nogil

cdef void c_leapfrog_step_nbody(
    CPotential *p, int half_ndim, double t, double dt,
    CPotential **pots, double *x_nbody_jm1, int nbody, int nbody_i,
    double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad
) nogil
