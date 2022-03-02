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


cdef class CFrameWrapper:
    """ Wrapper class for C implementation of reference frames. """

    cpdef init(self, list parameters):
        # save the array of parameters so it doesn't get garbage-collected
        self._params = np.array(parameters, dtype=np.float64)

    cpdef energy(self, double[:, ::1] w, double[::1] t):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrameType cf = self.cframe
        n, ndim = _validate_pos_arr(w)

        cdef double [::1] pot = np.zeros(n)
        if len(t) == 1:
            for i in range(n):
                pot[i] = frame_hamiltonian(&cf, t[0], &w[i, 0], ndim//2)
        else:
            for i in range(n):
                pot[i] = frame_hamiltonian(&cf, t[i], &w[i, 0], ndim//2)


        return np.array(pot)

    cpdef gradient(self, double[:, ::1] w, double[::1] t):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrameType cf = self.cframe
        n, ndim = _validate_pos_arr(w)

        cdef double[:, ::1] dH = np.zeros((n, ndim))
        if len(t) == 1:
            for i in range(n):
                frame_gradient(&cf, t[0], &w[i, 0], ndim//2, &dH[i, 0])
        else:
            for i in range(n):
                frame_gradient(&cf, t[i], &w[i, 0], ndim//2, &dH[i, 0])


        return np.array(dH)

    cpdef hessian(self, double[:, ::1] w, double[::1] t):
        """
        w should have shape (n, ndim).
        """
        cdef:
            int n, ndim, i
            CFrameType cf = self.cframe
        n, ndim = _validate_pos_arr(w)

        cdef double[:, :, ::1] d2H = np.zeros((n, ndim, ndim))
        if len(t) == 1:
            for i in range(n):
                frame_hessian(&cf, t[0], &w[i, 0], ndim//2, &d2H[i, 0, 0])
        else:
            for i in range(n):
                frame_hessian(&cf, t[i], &w[i, 0], ndim//2, &d2H[i, 0, 0])

        return np.array(d2H)

    def __reduce__(self):
        return (self.__class__, (list(self._params), ))


class CFrameBase(FrameBase):
    Wrapper = None

    def __init__(self, *args, units=None, **kwargs):
        super().__init__(*args, units=units, **kwargs)
        self._setup_wrapper()

    def _setup_wrapper(self):
        if self.Wrapper is None:
            raise ValueError("C potential wrapper class not defined for "
                             f"potential class {self.__class__}")

        # to support array parameters, but they get unraveled
        arrs = [np.atleast_1d(v.value).ravel()
                for v in self.parameters.values()]

        if len(arrs) > 0:
            self.c_parameters = np.concatenate(arrs)
        else:
            self.c_parameters = np.array([])

        self.c_instance = self.Wrapper(list(self.c_parameters))

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
