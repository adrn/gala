# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

# Project
from .core import PotentialBase, CompositePotential
from ..util import atleast_2d

cdef extern from "math.h":
    double sqrt(double x) nogil
    double fabs(double x) nogil

cdef extern from "stdint.h":
    ctypedef int intptr_t

class CPotentialBase(PotentialBase):
    """
    A base class for representing gravitational potentials with
    value, gradient, etc. functions implemented in C.

    TODO: better description here
    TODO: need tests of return shapes and handling 3d arrays!!!
    """

    def _value(self, q, t=0.):
        q = np.ascontiguousarray(q.T)
        return self.c_instance.value(q, t=t)

    def _gradient(self, q, t=0.):
        q = np.ascontiguousarray(q.T)
        try:
            return self.c_instance.gradient(q, t=t)
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "gradient function")

    def _density(self, q, t=0.):
        q = np.ascontiguousarray(q.T)
        try:
            return self.c_instance.density(q, t=t)
        except AttributeError,TypeError:
            # TODO: if no density function, should this numerically esimate
            #   the density?
            raise ValueError("Potential C instance has no defined "
                             "density function")

    def _hessian(self, q, t=0.):
        q = np.ascontiguousarray(q.T)
        try:
            return self.c_instance.hessian(q, t=t) # TODO: return shape?
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "Hessian function")

    # ----------------------------------------------------------
    # Overwrite the Python potential method to use Cython method
    def mass_enclosed(self, q, t=0.):
        """
        mass_enclosed(q)

        Estimate the mass enclosed within the given position by assuming the potential
        is spherical. This is not so good!

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the mass enclosed.
        """
        q = np.ascontiguousarray(atleast_2d(q, insert_axis=1).T)
        try:
            return self.c_instance.mass_enclosed(q, self.G, t=t)
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "mass_enclosed function")

# ==============================================================================

cdef class _CPotential:

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, d):
        # for k,v in d.items():
        #     setattr(self, k, v)
        pass

    def __getstate__(self):
        return None

    def __reduce__(self):
        return (self.__class__, tuple(self._parvec))

    cpdef value(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int norbits, ndim, i
        norbits = q.shape[0]
        ndim = q.shape[1]

        cdef double [::1] pot = np.zeros((norbits,))
        for i in range(norbits):
            pot[i] = self._value(t, &q[i,0])

        return np.array(pot)

    cdef public inline double _value(self, double t, double *r) nogil:
        return self.c_value(t, self._parameters, r)

    # -------------------------------------------------------------
    cpdef gradient(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int norbits, ndim, i
        norbits = q.shape[0]
        ndim = q.shape[1]

        cdef double [:,::1] grad = np.zeros((norbits,ndim))
        for i in range(norbits):
            self._gradient(t, &q[i,0], &grad[i,0])

        return np.array(grad).T

    cdef public inline void _gradient(self, double t, double *r, double *grad) nogil:
        self.c_gradient(t, self._parameters, r, grad)

    # -------------------------------------------------------------
    cpdef density(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int norbits, ndim, i
        norbits = q.shape[0]
        ndim = q.shape[1]

        cdef double [::1] pot = np.zeros((norbits,))
        for i in range(norbits):
            pot[i] = self._density(t, &q[i,0])

        return np.array(pot)

    cdef public inline double _density(self, double t, double *r) nogil:
        return self.c_density(t, self._parameters, r)

    # -------------------------------------------------------------
    cpdef hessian(self, double[:,::1] w):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int norbits, ndim, i
        norbits = w.shape[0]
        ndim = w.shape[1]

        cdef double [:,:,::1] hess = np.zeros((norbits,ndim,ndim))
        for i in range(norbits):
            self._hessian(&w[i,0], &hess[i,0,0])

        return np.array(hess) # TODO: this should be rollaxis

    cdef public void _hessian(self, double *w, double *hess) nogil:
        cdef int i,j
        for i in range(3):
            for j in range(3):
                hess[3*i+j] = 0.

    # -------------------------------------------------------------
    cpdef d_dr(self, double[:,::1] q, double G, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int norbits, i
        norbits = q.shape[0]

        cdef double [::1] epsilon = np.zeros(3)
        cdef double [::1] dr = np.zeros((norbits,))
        for i in range(norbits):
            dr[i] = self._d_dr(t, &q[i,0], &epsilon[0], G)
        return np.array(dr)

    cdef public double _d_dr(self, double t, double *q, double *epsilon, double Gee) nogil:
        cdef double h, r, dPhi_dr

        # Fractional step-size
        h = 0.01

        # Step-size for estimating radial gradient of the potential
        r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])

        for j in range(3):
            epsilon[j] = h * q[j]/r + q[j]
        dPhi_dr = self._value(t, epsilon)

        for j in range(3):
            epsilon[j] = h * q[j]/r - q[j]
        dPhi_dr -= self._value(t, epsilon)

        return dPhi_dr / (2.*h)

    cpdef d2_dr2(self, double[:,::1] q, double G, double t=0.):
        """
        d2_dr2(q, G, t=0.)

        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int norbits, i
        norbits = q.shape[0]

        cdef double [::1] epsilon = np.zeros(3)
        cdef double [::1] dr = np.zeros((norbits,))
        for i in range(norbits):
            dr[i] = self._d2_dr2(t, &q[i,0], &epsilon[0], G)
        return np.array(dr)

    cdef public double _d2_dr2(self, double t, double *q, double *epsilon, double Gee) nogil:
        cdef double h, r, d2Phi_dr2

        # Fractional step-size
        h = 0.01

        # Step-size for estimating radial gradient of the potential
        r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])

        for j in range(3):
            epsilon[j] = h * q[j]/r + q[j]
        d2Phi_dr2 = self._value(t, epsilon)

        d2Phi_dr2 -= 2*self._value(t, q)

        for j in range(3):
            epsilon[j] = h * q[j]/r - q[j]
        d2Phi_dr2 += self._value(t, epsilon)

        return d2Phi_dr2 / (h*h)

    cpdef mass_enclosed(self, double[:,::1] q, double G, double t=0.):
        cdef int norbits, i
        norbits = q.shape[1]

        cdef double [::1] epsilon = np.zeros(3)
        cdef double [::1] mass = np.zeros((norbits,))
        for i in range(norbits):
            mass[i] = self._mass_enclosed(t, &q[i,0], &epsilon[0], G)
        return np.array(mass)

    cdef public double _mass_enclosed(self, double t, double *q, double *epsilon, double Gee) nogil:
        cdef double r, dPhi_dr
        r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
        dPhi_dr = self._d_dr(t, &q[0], &epsilon[0], Gee)
        return fabs(r*r * dPhi_dr / Gee)

# ==============================================================================

# class CCompositePotential(CompositePotential, CPotentialBase):
#     """

#     TODO!

#     A baseclass for representing gravitational potentials. You must specify
#     a function that evaluates the potential value (func). You may also
#     optionally add a function that computes derivatives (gradient), and a
#     function to compute second derivatives (the Hessian) of the potential.

#     Parameters
#     ----------
#     TODO
#     """

#     def __init__(self, **kwargs):
#         super(CCompositePotential, self).__init__(**kwargs)

#         self.c_instance = _CCompositePotential([v.c_instance for v in self.values()],
#                                                n=len(self))  # number of components

# from cpython cimport PyObject
# cdef class _CCompositePotential: #(_CPotential):

#     def __init__(self, cobjs, n):
#         """ Need a list of instances of _CPotential classes """
#         cdef int i

#         self.n = n
#         self.cpotentials = np.array(cobjs, dtype=_CPotential)
#         self.pointers = np.zeros(self.n, dtype=np.int32)
#         self.param_pointers = np.zeros(self.n, dtype=np.int32)

#         for i in range(n):
#             self.pointers[i] = <intptr_t>&(self.cpotentials[i].c_value)
#             self.param_pointers[i] = <intptr_t>&(self.cpotentials[i]._parameters[0])

#         self._pointers = &(self.pointers[0])
#         self._param_pointers = &(self.pointers[0])

#         composite_value

#     # def __setstate__(self, d):
#     #     # for k,v in d.items():
#     #     #     setattr(self, k, v)
#     #     pass

#     # def __getstate__(self):
#     #     return None

#     cdef public double _value(self, double* q) nogil:
#         # whoa this is some whack cython wizardry right here
#         #   (stolen from stackoverflow)
#         cdef int i
#         cdef double v = 0.
#         for i in range(self.n):
#             v += (<valuefunc>(self._pointers[i]))(<double*>(self._param_pointers[i]), &q[0])
#         return v

#     cpdef value(self, double[:,::1] q):
#         cdef int norbits, ndim, k
#         norbits = q.shape[0]
#         ndim = q.shape[1]

#         cdef double [::1] pot = np.zeros((norbits,))
#         for k in range(norbits):
#             pot[k] = self._value(&q[k,0])

#         return np.array(pot)

#     # cdef public void _gradient(self, double *r, double *grad) nogil:
#     #     # whoa this is some whack cython wizardry right here
#     #     #   (stolen from stackoverflow)
#     #     cdef double v = 0.

#     #     for i in range(self.ninstances):
#     #         (<_CPotential>(self.obj_list[i]))._gradient(r, grad)

#     # cdef public void _hessian(self, double *w, double *hess) nogil:
#     #     # whoa this is some whack cython wizardry right here
#     #     #   (stolen from stackoverflow)
#     #     for i in range(self.ninstances):
#     #         (<_CPotential>(self.obj_list[i]))._hessian(w, hess)

#     # cdef public double _mass_enclosed(self, double *q, double *epsilon, double Gee) nogil:
#     #     cdef double mm = 0.
#     #     for i in range(self.ninstances):
#     #         mm += (<_CPotential>(self.obj_list[i]))._mass_enclosed(q, epsilon, Gee)
#     #     return mm

