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
from astropy.extern import six
from astropy.utils import InheritDocstrings
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

from libc.stdio cimport printf

# Project
from .core import PotentialBase, CompositePotential
from ..util import atleast_2d
from ..units import DimensionlessUnitSystem

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
        int n_dim
        densityfunc density[MAX_N_COMPONENTS]
        valuefunc value[MAX_N_COMPONENTS]
        gradientfunc gradient[MAX_N_COMPONENTS]
        int n_params[MAX_N_COMPONENTS]
        double *parameters[MAX_N_COMPONENTS]

    double c_value(CPotential *p, double t, double *q) nogil
    double c_density(CPotential *p, double t, double *q) nogil
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil

    double c_d_dr(CPotential *p, double t, double *q, double *epsilon) nogil
    double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) nogil
    double c_mass_enclosed(CPotential *p, double t, double *q, double G, double *epsilon) nogil

__all__ = ['CPotentialBase']

cdef class CPotentialWrapper:
    """
    Wrapper class for C implementation of potentials. At the C layer, potentials
    are effectively struct's that maintain pointers to functions specific to a
    given potential. This provides a Cython wrapper around this C implementation.
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

        cdef double [:,::1] grad = np.zeros((norbits, ndim))
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

    # Second order functionality

    cpdef d_dr(self, double[:,::1] q, double G, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef:
            int i
            int norbits = q.shape[0]
            double [::1] epsilon = np.zeros(3, dtype=np.float64)
            double [::1] dr = np.zeros(norbits, dtype=np.float64)

        for i in range(norbits):
            dr[i] = c_d_dr(&(self.cpotential), t, &q[i,0], &epsilon[0])

        return np.array(dr)

    cpdef d2_dr2(self, double[:,::1] q, double G, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef:
            int i
            int norbits = q.shape[0]
            double [::1] epsilon = np.zeros(3, dtype=np.float64)
            double [::1] dr2 = np.zeros(norbits, dtype=np.float64)

        for i in range(norbits):
            dr2[i] = c_d2_dr2(&(self.cpotential), t, &q[i,0], &epsilon[0])

        return np.array(dr2)

    cpdef mass_enclosed(self, double[:,::1] q, double G, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef:
            int i
            int norbits = q.shape[0]
            double [::1] epsilon = np.zeros(3, dtype=np.float64)
            double [::1] mass = np.zeros(norbits, dtype=np.float64)

        for i in range(norbits):
            mass[i] = c_mass_enclosed(&(self.cpotential), t, &q[i,0], G, &epsilon[0])

        return np.array(mass)

    # For pickling in Python 2
    def __reduce__(self):
        return (self.__class__, (self._params[0], list(self._params[1:])))

class CPotentialBase(PotentialBase):
    """
    A baseclass for defining gravitational potentials implemented in C.
    """

    def __init__(self, parameters, units, Wrapper=None):
        super(CPotentialBase, self).__init__(parameters, units=units)

        c_params = []
        for k,v in self.parameters.items():
            c_params.append(self.parameters[k].value)
        self.c_parameters = np.array(c_params)

        if Wrapper is None:
            # magic to set the c_instance attribute based on the name of the class
            wrapper_name = '{}Wrapper'.format(self.__class__.__name__.replace('Potential', ''))

            from .builtin import cybuiltin
            Wrapper = getattr(cybuiltin, wrapper_name)

        self.c_instance = Wrapper(self.G, self.c_parameters)


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
            return self.c_instance.gradient(q, t=t).T.reshape(sh)
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
        mass_enclosed(q, t)

        Estimate the mass enclosed within the given position by assuming the potential
        is spherical. This is not so good!

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the mass enclosed.
        """
        if isinstance(self.units, DimensionlessUnitSystem):
            raise ValueError("No units specified when creating potential object.")

        q = atleast_2d(q, insert_axis=1)
        sh = q.shape
        q = np.ascontiguousarray(q.reshape(sh[0],np.prod(sh[1:])).T)
        try:
            menc = self.c_instance.mass_enclosed(q, self.G, t=t)
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "mass_enclosed function")
        return menc.reshape(sh[1:]) * self.units['mass']
