# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

__all__ = ['CFrameBase']

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

from ...dynamics import CartesianPhaseSpacePosition

cdef extern from "src/funcdefs.h":
    ctypedef double (*valuefunc)(double t, double *pars, double *q) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, double *hess) nogil

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        pass

    double frame_hamiltonian(CFrame *fr, double t, double *qp) nogil
    void frame_gradient(CFrame *fr, double t, double *qp, double *dH) nogil
    void frame_hessian(CFrame *fr, double t, double *qp, double *d2H) nogil

cdef class CFrameWrapper:
    """ Wrapper class for C implementation of reference frames. """

    cpdef _validate_w(self, double[:,::1] w):
        if w.ndim != 2:
            raise ValueError("Phase-space coordinate array w must have 2 dimensions")

        return w.shape[0], w.shape[1]

    cpdef energy(self, double[:,::1] w, double t=0.):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrame cf = self.cframe

        n,ndim = self._validate_w(w)

        cdef double [::1] pot = np.zeros((n,))
        for i in range(n):
            pot[i] = frame_hamiltonian(&cf, t, &w[i,0])

        return np.array(pot)

    cpdef gradient(self, double[:,::1] w, double t=0.):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrame cf = self.cframe
        n,ndim = self._validate_w(w)

        cdef double[:,::1] dH = np.zeros((n, ndim))
        for i in range(n):
            frame_gradient(&cf, t, &w[i,0], &dH[i,0])

        return np.array(dH)

    cpdef hessian(self, double[:,::1] w, double t=0.):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrame cf = self.cframe
        n,ndim = self._validate_w(w)

        cdef double[:,:,::1] d2H = np.zeros((n, ndim, ndim))

        for i in range(n):
            frame_hessian(&cf, t, &w[i,0], &d2H[i,0,0])

        return np.array(d2H)

# TODO: make sure this doesn't appear in docs - Frames are really only used internally
class CFrameBase(object):

    def __init__(self, c_instance):
        self.c_instance = c_instance

    def _energy(self, w, t=0.):
        orig_shp = w.shape
        w = np.ascontiguousarray(w.reshape(orig_shp[0], -1).T)
        return self.c_instance.energy(w, t=t).reshape(orig_shp[1:])

    def _gradient(self, w, t=0.):
        orig_shp = w.shape
        w = np.ascontiguousarray(w.reshape(orig_shp[0], -1).T)
        return self.c_instance.gradient(w, t=t).T.reshape(orig_shp)

    def _hessian(self, w, t=0.):
        orig_shp = w.shape
        w = np.ascontiguousarray(w.reshape(orig_shp[0], -1).T)
        hess = self.c_instance.hessian(w, t=t)
        return np.moveaxis(hess, 0, -1).reshape((orig_shp[0], orig_shp[0]) + orig_shp[1:])

