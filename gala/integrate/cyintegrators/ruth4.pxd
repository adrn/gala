# cython: language_level=3

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        pass

cdef void c_ruth4_step(CPotential *p, int ndim, double t, double dt,
                       double *cs, double *ds,
                       double *w, double *grad) nogil
