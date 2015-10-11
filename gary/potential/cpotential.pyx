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

cdef extern from "math.h":
    double sqrt(double x) nogil
    double fabs(double x) nogil

cdef extern from "stdint.h":
    ctypedef int intptr_t

class CPotentialBase(PotentialBase):
    """
    A base class for representing gravitational potentials implemented in Cython.

    You must specify a function that evaluates the potential value (func). You may also
    optionally add a function that computes derivatives (gradient), and a
    function to compute second derivatives (the Hessian) of the potential.

    Parameters
    ----------
    func : function
        A function that computes the value of the potential.
    gradient : function (optional)
        A function that computes the first derivatives (gradient) of the potential.
    hessian : function (optional)
        A function that computes the second derivatives (Hessian) of the potential.
    parameters : dict (optional)
        Any extra parameters that the functions (func, gradient, hessian)
        require. All functions must take the same parameters.

    """

    def value(self, q, t=0.):
        """
        value(q, t=0)

        Compute the value of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the value of the potential.
        t : numeric (optional)
            The time.
        """
        return self.c_instance.value(np.ascontiguousarray(np.atleast_2d(q)), t=t)

    def gradient(self, q, t=0.):
        """
        gradient(q, t=0)

        Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the gradient.
        t : numeric (optional)
            The time.
        """
        try:
            return self.c_instance.gradient(np.ascontiguousarray(np.atleast_2d(q)), t=t)
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "gradient function")

    def density(self, q, t=0.):
        """
        density(q, t=0)

        Compute the value of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the value of the potential.
        t : numeric (optional)
            The time.
        """
        return self.c_instance.density(np.ascontiguousarray(np.atleast_2d(q)), t=t)

    def hessian(self, q):
        """
        hessian(q)

        Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the Hessian.
        """
        try:
            return self.c_instance.hessian(np.ascontiguousarray(np.atleast_2d(q)))
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "Hessian function")

    # ----------------------------
    # Functions of the derivatives
    # ----------------------------
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
        try:
            return self.c_instance.mass_enclosed(np.ascontiguousarray(np.atleast_2d(q)), self.G, t=t)
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
        # args = (self.G, self.m, self.c)
        # return (_HernquistPotential, tuple(self._parvec))
        return (self.__class__, tuple(self._parvec))

    cpdef value(self, double[:,::1] q, double t=0.):
        cdef int nparticles, ndim, k
        nparticles = q.shape[0]
        ndim = q.shape[1]

        cdef double [::1] pot = np.zeros((nparticles,))
        for k in range(nparticles):
            pot[k] = self._value(t, &q[k,0])

        return np.array(pot)

    cdef public inline double _value(self, double t, double *r) nogil:
        return self.c_value(t, self._parameters, r)

    # -------------------------------------------------------------
    cpdef gradient(self, double[:,::1] q, double t=0.):
        cdef int nparticles, ndim, k
        nparticles = q.shape[0]
        ndim = q.shape[1]

        cdef double [:,::1] grad = np.zeros((nparticles,ndim))
        for k in range(nparticles):
            self._gradient(t, &q[k,0], &grad[k,0])

        return np.array(grad)

    cdef public inline void _gradient(self, double t, double *r, double *grad) nogil:
        self.c_gradient(t, self._parameters, r, grad)

    # -------------------------------------------------------------
    cpdef density(self, double[:,::1] q, double t=0.):
        cdef int nparticles, ndim, k
        nparticles = q.shape[0]
        ndim = q.shape[1]

        cdef double [::1] pot = np.zeros((nparticles,))
        for k in range(nparticles):
            pot[k] = self._density(t, &q[k,0])

        return np.array(pot)

    cdef public inline double _density(self, double t, double *r) nogil:
        return self.c_density(t, self._parameters, r)

    # -------------------------------------------------------------
    cpdef hessian(self, double[:,::1] w):
        cdef int nparticles, ndim, k
        nparticles = w.shape[0]
        ndim = w.shape[1]

        cdef double [:,:,::1] hess = np.zeros((nparticles,ndim,ndim))
        for k in range(nparticles):
            self._hessian(&w[k,0], &hess[k,0,0])

        return np.array(hess)

    cdef public void _hessian(self, double *w, double *hess) nogil:
        cdef int i,j
        for i in range(3):
            for j in range(3):
                hess[3*i+j] = 0.

    # -------------------------------------------------------------
    cpdef d_dr(self, double[:,::1] q, double G, double t=0.):
        cdef int nparticles, k
        nparticles = q.shape[0]

        cdef double [::1] epsilon = np.zeros(3)
        cdef double [::1] dr = np.zeros((nparticles,))
        for k in range(nparticles):
            dr[k] = self._d_dr(t, &q[k,0], &epsilon[0], G)
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

    cpdef mass_enclosed(self, double[:,::1] q, double G, double t=0.):
        cdef int nparticles, k
        nparticles = q.shape[0]

        cdef double [::1] epsilon = np.zeros(3)
        cdef double [::1] mass = np.zeros((nparticles,))
        for k in range(nparticles):
            mass[k] = self._mass_enclosed(t, &q[k,0], &epsilon[0], G)
        return np.array(mass)

    cdef public double _mass_enclosed(self, double t, double *q, double *epsilon, double Gee) nogil:
        cdef double r, dPhi_dr
        dPhi_dr = self._d_dr(t, &q[0], &epsilon[0], Gee)
        return dPhi_dr / (2.*h)

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
#         cdef int nparticles, ndim, k
#         nparticles = q.shape[0]
#         ndim = q.shape[1]

#         cdef double [::1] pot = np.zeros((nparticles,))
#         for k in range(nparticles):
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

