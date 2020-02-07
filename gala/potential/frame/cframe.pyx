# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

__all__ = ['CFrameBase']


# Third-party
import numpy as np
cimport numpy as np
np.import_array()

from .core import FrameBase
from ..potential.cpotential import _validate_pos_arr
from ..potential.cpotential cimport energyfunc, gradientfunc, hessianfunc
from ...dynamics import PhaseSpacePosition


cdef class CFrameWrapper:
    """ Wrapper class for C implementation of reference frames. """

    cpdef energy(self, double[:,::1] w, double[::1] t):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrame cf = self.cframe
        n,ndim = _validate_pos_arr(w)

        cdef double [::1] pot = np.zeros(n)
        if len(t) == 1:
            for i in range(n):
                pot[i] = frame_hamiltonian(&cf, t[0], &w[i,0], ndim//2)
        else:
            for i in range(n):
                pot[i] = frame_hamiltonian(&cf, t[i], &w[i,0], ndim//2)


        return np.array(pot)

    cpdef gradient(self, double[:,::1] w, double[::1] t):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrame cf = self.cframe
        n,ndim = _validate_pos_arr(w)

        cdef double[:,::1] dH = np.zeros((n, ndim))
        if len(t) == 1:
            for i in range(n):
                frame_gradient(&cf, t[0], &w[i,0], ndim//2, &dH[i,0])
        else:
            for i in range(n):
                frame_gradient(&cf, t[i], &w[i,0], ndim//2, &dH[i,0])


        return np.array(dH)

    cpdef hessian(self, double[:,::1] w, double[::1] t):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrame cf = self.cframe
        n,ndim = _validate_pos_arr(w)

        cdef double[:,:,::1] d2H = np.zeros((n, ndim, ndim))
        if len(t) == 1:
            for i in range(n):
                frame_hessian(&cf, t[0], &w[i,0], ndim//2, &d2H[i,0,0])
        else:
            for i in range(n):
                frame_hessian(&cf, t[i], &w[i,0], ndim//2, &d2H[i,0,0])

        return np.array(d2H)


class CFrameBase(FrameBase):

    def __init__(self, Wrapper, parameters, units, ndim=3):
        self.units = self._validate_units(units)
        self.parameters = self._prepare_parameters(parameters, self.units)
        self.c_parameters = np.ravel([v.value for v in self.parameters.values()])
        self.c_instance = Wrapper(*self.c_parameters)

        self.ndim = ndim

    def __str__(self):
        return self.__class__.__name__

    def _energy(self, q, t):
        return self.c_instance.energy(q, t=t)

    def _gradient(self, q, t):
        return self.c_instance.gradient(q, t=t)

    def _density(self, q, t):
        return self.c_instance.density(q, t=t)

    def _hessian(self, q, t):
        return self.c_instance.hessian(q, t=t)
