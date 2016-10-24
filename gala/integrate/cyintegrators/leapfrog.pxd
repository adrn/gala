cdef extern from "src/cframe.h":
    ctypedef struct CFrame:
        pass

cdef extern from "src/cpotential.h":
    ctypedef struct CPotential:
        pass

cdef void c_init_velocity(CPotential *p, CFrame *fr, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil

cdef void c_leapfrog_step(CPotential *p, CFrame *fr, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil
