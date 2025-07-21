# cython: language_level=3
# cython: language=c++

from ...potential.potential.cpotential cimport CPotential

cdef void c_ruth4_step(CPotential *p, int ndim, double t, double dt,
                       double *cs, double *ds,
                       double *w, double *grad) nogil
