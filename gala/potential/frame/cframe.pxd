# cython: language_level=3

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrameType:
        pass

    double frame_hamiltonian(CFrameType *fr, double t, double *qp, int n_dim) nogil
    void frame_gradient(CFrameType *fr, double t, double *qp, int n_dim, double *dH) nogil
    void frame_hessian(CFrameType *fr, double t, double *qp, int n_dim, double *d2H) nogil

cdef class CFrameWrapper:
    cdef CFrameType cframe
    cdef double[::1] _params
    cpdef init(self, list parameters)
    cpdef energy(self, double[:,::1] w, double[::1] t)
    cpdef gradient(self, double[:,::1] w, double[::1] t)
    cpdef hessian(self, double[:,::1] w, double[::1] t)
