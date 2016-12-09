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

from .core import FrameBase
from ..potential.cpotential import _validate_pos_arr
from ...dynamics import CartesianPhaseSpacePosition

cdef extern from "src/funcdefs.h":
    ctypedef double (*energyfunc)(double t, double *pars, double *q, int n_dim) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, int n_dim, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, int n_dim, double *hess) nogil

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        pass

    double frame_hamiltonian(CFrame *fr, double t, double *qp, int n_dim) nogil
    void frame_gradient(CFrame *fr, double t, double *qp, int n_dim, double *dH) nogil
    void frame_hessian(CFrame *fr, double t, double *qp, int n_dim, double *d2H) nogil

cdef class CFrameWrapper:
    """ Wrapper class for C implementation of reference frames. """

    cpdef energy(self, double[:,::1] w, double t=0.):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrame cf = self.cframe
        n,ndim = _validate_pos_arr(w)

        cdef double [::1] pot = np.zeros(n)
        for i in range(n):
            pot[i] = frame_hamiltonian(&cf, t, &w[i,0], ndim//2)

        return np.array(pot)

    cpdef gradient(self, double[:,::1] w, double t=0.):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrame cf = self.cframe
        n,ndim = _validate_pos_arr(w)

        cdef double[:,::1] dH = np.zeros((n, ndim))
        for i in range(n):
            frame_gradient(&cf, t, &w[i,0], ndim//2, &dH[i,0])

        return np.array(dH)

    cpdef hessian(self, double[:,::1] w, double t=0.):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrame cf = self.cframe
        n,ndim = _validate_pos_arr(w)

        cdef double[:,:,::1] d2H = np.zeros((n, ndim, ndim))

        for i in range(n):
            frame_hessian(&cf, t, &w[i,0], ndim//2, &d2H[i,0,0])

        return np.array(d2H)

# TODO: make sure this doesn't appear in docs - Frames are really only used internally
class CFrameBase(FrameBase):

    def __init__(self, Wrapper, parameters, units, parameter_physical_types=None, ndim=3):
        self.units = self._validate_units(units)

        if parameter_physical_types is None:
            parameter_physical_types = dict()
        self._ptypes = parameter_physical_types

        self.parameters = self._prepare_parameters(parameters, self._ptypes, self.units)
        self.c_parameters = np.ravel([v.value for v in self.parameters.values()])
        self.c_instance = Wrapper(*self.c_parameters)

        self.ndim = ndim

    def __str__(self):
        return self.__class__.__name__

    def _energy(self, q, t=0.):
        return self.c_instance.energy(q, t=t)

    def _gradient(self, q, t=0.):
        return self.c_instance.gradient(q, t=t)

    def _density(self, q, t=0.):
        return self.c_instance.density(q, t=t)

    def _hessian(self, q, t=0.):
        return self.c_instance.hessian(q, t=t)
