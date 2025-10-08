# cython: language_level=3
# cython: language=c++

from ..potential.cpotential cimport energyfunc, gradientfunc, hessianfunc

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrameType:
        energyfunc energy
        gradientfunc gradient
        hessianfunc hessian

        int n_params
        double *parameters

    double frame_hamiltonian(CFrameType *fr, double t, double *qp, int n_dim) except + nogil
    void frame_gradient(CFrameType *fr, double t, double *qp, int n_dim, size_t N, double *dH) except + nogil
    void frame_hessian(CFrameType *fr, double t, double *qp, int n_dim, double *d2H) except + nogil

cdef class CFrameWrapper:
    cdef CFrameType cframe
    cdef double[::1] _params
    cpdef init(self, list parameters)
    cpdef energy(self, double[:,::1] w, double[::1] t)
    cpdef gradient(self, double[:,::1] w, double[::1] t)
    cpdef hessian(self, double[:,::1] w, double[::1] t)
