from ...potential.potential.cpotential cimport CPotentialWrapper
from ...potential.frame.cframe cimport CFrameWrapper

cdef class BaseStreamDF:

    cdef double _lead
    cdef double _trail
    cdef CPotentialWrapper _potential
    cdef CFrameWrapper _frame
    cdef double _G
    cdef dict extra_kwargs

    cdef void get_rj_vj_R(self, double *prog_x, double *prog_v,
                          double prog_m, double t,
                          double *rj, double *vj, double[:, ::1] R)

    cdef void transform_from_sat(self, double[:, ::1] R,
                                 double *x, double *v,
                                 double *prog_x, double *prog_v,
                                 double *out_x, double *out_v)

    cpdef _sample(self, double[:, ::1] prog_x, double[:, ::1] prog_v,
                  double[::1] prog_t, double[::1] prog_m, int[::1] nparticles,
                  dict extra_kwargs)
