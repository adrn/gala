# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

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

cdef extern from "src/cpotential.h":
    enum:
        MAX_N_COMPONENTS = 16

    ctypedef double (*densityfunc)(double t, double *pars, double *q) nogil
    ctypedef double (*valuefunc)(double t, double *pars, double *q) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil

    ctypedef struct CPotential:
        int n_components
        densityfunc density[MAX_N_COMPONENTS]
        valuefunc value[MAX_N_COMPONENTS]
        gradientfunc gradient[MAX_N_COMPONENTS]
        int n_params[MAX_N_COMPONENTS]
        double *parameters[MAX_N_COMPONENTS]

    double c_value(CPotential *p, double t, double *q) nogil
    double c_density(CPotential *p, double t, double *q) nogil
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil

__all__ = ['CPotentialBase']

cdef class CPotentialWrapper:
    """
    Wrapper class for C struct potential container.
    """

    cpdef value(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int norbits, ndim, i

        if q.ndim != 2:
            raise ValueError("Coordinate array q must have 2 dimensions")

        norbits = q.shape[0]
        ndim = q.shape[1]

        cdef double [::1] pot = np.zeros((norbits,))
        for i in range(norbits):
            pot[i] = c_value(&(self.cpotential), t, &q[i,0])

        return np.array(pot)

    cpdef density(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int norbits, ndim, i

        if q.ndim != 2:
            raise ValueError("Coordinate array q must have 2 dimensions")

        norbits = q.shape[0]
        ndim = q.shape[1]

        cdef double [::1] dens = np.zeros((norbits,))
        for i in range(norbits):
            dens[i] = c_density(&(self.cpotential), t, &q[i,0])

        return np.array(dens)

    cpdef gradient(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int norbits, ndim, i

        if q.ndim != 2:
            raise ValueError("Coordinate array q must have 2 dimensions")

        norbits = q.shape[0]
        ndim = q.shape[1]

        cdef double [:,::1] grad = np.zeros(q.shape)
        for i in range(norbits):
            c_gradient(&(self.cpotential), t, &q[i,0], &grad[i,0])

        return np.array(grad)

    # cpdef hessian(self, double[:,::1] q, double t=0.):
    #     """
    #     CAUTION: Interpretation of axes is different here! We need the
    #     arrays to be C ordered and easy to iterate over, so here the
    #     axes are (norbits, ndim).
    #     """
    #     cdef int norbits, ndim, i

    #     if q.ndim != 2:
    #         raise ValueError("Coordinate array q must have 2 dimensions")

    #     norbits = q.shape[0]
    #     ndim = q.shape[1]

    #     cdef double [::1] hess = np.zeros(q.shape + (ndim,)))
    #     for i in range(norbits):
    #         c_hessian(&(self.cpotential), t, &q[i,0], &hess[i,0,0])

    #     return np.array(hess)

class CPotentialBase(PotentialBase):
    """
    A base class for representing gravitational potentials with
    value, gradient, etc. functions implemented in C.

    TODO: better description here
    """

    def __init__(self, parameters, units):
        super(CPotentialBase, self).__init__(parameters, units=units)

        c_params = []
        for k,v in self.parameters.items():
            c_params.append(self.parameters[k].value)
        self.c_parameters = np.array(c_params)

    def _value(self, q, t=0.):
        sh = q.shape
        q = np.ascontiguousarray(q.reshape(sh[0],np.prod(sh[1:])).T)
        return self.c_instance.value(q, t=t).reshape(sh[1:])

    def _density(self, q, t=0.):
        sh = q.shape
        q = np.ascontiguousarray(q.reshape(sh[0],np.prod(sh[1:])).T)
        try:
            return self.c_instance.density(q, t=t).reshape(sh[1:])
        except AttributeError,TypeError:
            # TODO: if no density function, should this numerically esimate
            #   the density?
            raise ValueError("Potential C instance has no defined "
                             "density function")

    def _gradient(self, q, t=0.):
        sh = q.shape
        q = np.ascontiguousarray(q.reshape(sh[0],np.prod(sh[1:])).T)
        try:
            return self.c_instance.gradient(q, t=t).reshape(sh)
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "gradient function")

    def _hessian(self, q, t=0.):
        sh = q.shape
        q = np.ascontiguousarray(q.reshape(sh[0],np.prod(sh[1:])).T)
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
        if self.units is None:
            raise ValueError("No units specified when creating potential object.")

        q = atleast_2d(q, insert_axis=1)
        sh = q.shape
        q = np.ascontiguousarray(q.reshape(sh[0],np.prod(sh[1:])).T)
        try:
            menc = self.c_instance.mass_enclosed(q, self.G, t=t)
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "mass_enclosed function")
        return menc.reshape(sh[1:])

# ==============================================================================

# cdef class _CPotential:

#     def __init__(self, *args, **kwargs):
#         pass

#     def __setstate__(self, d):
#         # for k,v in d.items():
#         #     setattr(self, k, v)
#         pass

#     def __getstate__(self):
#         return None

#     def __reduce__(self):
#         return (self.__class__, tuple(self._parvec))

#     # -------------------------------------------------------------
#     cpdef d_dr(self, double[:,::1] q, double G, double t=0.):
#         """
#         CAUTION: Interpretation of axes is different here! We need the
#         arrays to be C ordered and easy to iterate over, so here the
#         axes are (norbits, ndim).
#         """
#         cdef int norbits, i
#         norbits = q.shape[0]

#         cdef double [::1] epsilon = np.zeros(3)
#         cdef double [::1] dr = np.zeros((norbits,))
#         for i in range(norbits):
#             dr[i] = self._d_dr(t, &q[i,0], &epsilon[0], G)
#         return np.array(dr)

#     cdef public double _d_dr(self, double t, double *q, double *epsilon, double Gee) nogil:
#         cdef double h, r, dPhi_dr

#         # Fractional step-size
#         h = 0.01

#         # Step-size for estimating radial gradient of the potential
#         r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])

#         for j in range(3):
#             epsilon[j] = h * q[j]/r + q[j]
#         dPhi_dr = self._value(t, epsilon)

#         for j in range(3):
#             epsilon[j] = h * q[j]/r - q[j]
#         dPhi_dr -= self._value(t, epsilon)

#         return dPhi_dr / (2.*h)

#     cpdef d2_dr2(self, double[:,::1] q, double G, double t=0.):
#         """
#         d2_dr2(q, G, t=0.)

#         CAUTION: Interpretation of axes is different here! We need the
#         arrays to be C ordered and easy to iterate over, so here the
#         axes are (norbits, ndim).
#         """
#         cdef int norbits, i
#         norbits = q.shape[0]

#         cdef double [::1] epsilon = np.zeros(3)
#         cdef double [::1] dr = np.zeros((norbits,))
#         for i in range(norbits):
#             dr[i] = self._d2_dr2(t, &q[i,0], &epsilon[0], G)
#         return np.array(dr)

#     cdef public double _d2_dr2(self, double t, double *q, double *epsilon, double Gee) nogil:
#         cdef double h, r, d2Phi_dr2

#         # Fractional step-size
#         h = 0.01

#         # Step-size for estimating radial gradient of the potential
#         r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])

#         for j in range(3):
#             epsilon[j] = h * q[j]/r + q[j]
#         d2Phi_dr2 = self._value(t, epsilon)

#         d2Phi_dr2 -= 2*self._value(t, q)

#         for j in range(3):
#             epsilon[j] = h * q[j]/r - q[j]
#         d2Phi_dr2 += self._value(t, epsilon)

#         return d2Phi_dr2 / (h*h)

#     cpdef mass_enclosed(self, double[:,::1] q, double G, double t=0.):
#         cdef int norbits, i
#         norbits = q.shape[0] # follows Cython axis convention

#         cdef double [::1] epsilon = np.zeros(3)
#         cdef double [::1] mass = np.zeros((norbits,))
#         for i in range(norbits):
#             mass[i] = self._mass_enclosed(t, &q[i,0], &epsilon[0], G)
#         return np.array(mass)

#     cdef public double _mass_enclosed(self, double t, double *q, double *epsilon, double Gee) nogil:
#         cdef double r, dPhi_dr
#         r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])
#         dPhi_dr = self._d_dr(t, &q[0], &epsilon[0], Gee)
#         return fabs(r*r * dPhi_dr / Gee)
