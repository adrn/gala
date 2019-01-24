# cython: language_level=3

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        pass

    double frame_hamiltonian(CFrame *fr, double t, double *qp, int n_dim) nogil
    void frame_gradient(CFrame *fr, double t, double *qp, int n_dim, double *dH) nogil
    void frame_hessian(CFrame *fr, double t, double *qp, int n_dim, double *d2H) nogil

cdef class CFrameWrapper:
    cdef CFrame cframe
    cdef double[::1] _params
    cpdef energy(self, double[:,::1] w, double[::1] t)
    cpdef gradient(self, double[:,::1] w, double[::1] t)
    cpdef hessian(self, double[:,::1] w, double[::1] t)
