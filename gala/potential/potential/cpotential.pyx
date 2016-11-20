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
import sys
import warnings

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
from ...util import atleast_2d
from ...units import DimensionlessUnitSystem

cdef extern from "math.h":
    double sqrt(double x) nogil
    double fabs(double x) nogil

cdef extern from "src/funcdefs.h":
    ctypedef double (*densityfunc)(double t, double *pars, double *q) nogil
    ctypedef double (*energyfunc)(double t, double *pars, double *q) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, double *hess) nogil

cdef extern from "potential/src/cpotential.h":
    enum:
        MAX_N_COMPONENTS = 16

    ctypedef struct CPotential:
        int n_components
        int n_dim
        densityfunc density[MAX_N_COMPONENTS]
        energyfunc value[MAX_N_COMPONENTS]
        gradientfunc gradient[MAX_N_COMPONENTS]
        hessianfunc hessian[MAX_N_COMPONENTS]
        int n_params[MAX_N_COMPONENTS]
        double *parameters[MAX_N_COMPONENTS]

    double c_potential(CPotential *p, double t, double *q) nogil
    double c_density(CPotential *p, double t, double *q) nogil
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil
    void c_hessian(CPotential *p, double t, double *q, double *hess) nogil

    double c_d_dr(CPotential *p, double t, double *q, double *epsilon) nogil
    double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) nogil
    double c_mass_enclosed(CPotential *p, double t, double *q, double G, double *epsilon) nogil

__all__ = ['CPotentialBase']

cpdef _validate_pos_arr(double[:,::1] arr):
    if arr.ndim != 2:
        raise ValueError("Phase-space coordinate array must have 2 dimensions")
    return arr.shape[0], arr.shape[1]

cdef class CPotentialWrapper:
    """
    Wrapper class for C implementation of potentials. At the C layer, potentials
    are effectively struct's that maintain pointers to functions specific to a
    given potential. This provides a Cython wrapper around this C implementation.
    """

    cpdef energy(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n,ndim = _validate_pos_arr(q)

        cdef double [::1] pot = np.zeros(n)
        for i in range(n):
            pot[i] = c_potential(&(self.cpotential), t, &q[i,0])

        return np.array(pot)

    cpdef density(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n,ndim = _validate_pos_arr(q)

        cdef double [::1] dens = np.zeros(n)
        for i in range(n):
            dens[i] = c_density(&(self.cpotential), t, &q[i,0])

        return np.array(dens)

    cpdef gradient(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n,ndim = _validate_pos_arr(q)

        cdef double[:,::1] grad = np.zeros((n, ndim))
        for i in range(n):
            c_gradient(&(self.cpotential), t, &q[i,0], &grad[i,0])

        return np.array(grad)

    cpdef hessian(self, double[:,::1] q, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n,ndim = _validate_pos_arr(q)

        cdef double[:,:,::1] hess = np.zeros((n, ndim, ndim))

        for i in range(n):
            c_hessian(&(self.cpotential), t, &q[i,0], &hess[i,0,0])

        return np.array(hess)

    # ------------------------------------------------------------------------
    # Other functionality
    #
    cpdef d_dr(self, double[:,::1] q, double G, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n,ndim = _validate_pos_arr(q)

        cdef double [::1] dr = np.zeros(n, dtype=np.float64)
        cdef double [::1] epsilon = np.zeros(3, dtype=np.float64)

        for i in range(n):
            dr[i] = c_d_dr(&(self.cpotential), t, &q[i,0], &epsilon[0])

        return np.array(dr)

    cpdef d2_dr2(self, double[:,::1] q, double G, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n,ndim = _validate_pos_arr(q)

        cdef double [::1] dr2 = np.zeros(n, dtype=np.float64)
        cdef double [::1] epsilon = np.zeros(3, dtype=np.float64)

        for i in range(n):
            dr2[i] = c_d2_dr2(&(self.cpotential), t, &q[i,0], &epsilon[0])

        return np.array(dr2)

    cpdef mass_enclosed(self, double[:,::1] q, double G, double t=0.):
        """
        CAUTION: Interpretation of axes is different here! We need the
        arrays to be C ordered and easy to iterate over, so here the
        axes are (norbits, ndim).
        """
        cdef int n, ndim, i
        n,ndim = _validate_pos_arr(q)

        cdef double [::1] mass = np.zeros(n, dtype=np.float64)
        cdef double [::1] epsilon = np.zeros(3, dtype=np.float64)

        for i in range(n):
            mass[i] = c_mass_enclosed(&(self.cpotential), t, &q[i,0], G, &epsilon[0])

        return np.array(mass)

    # For pickling in Python 2
    def __reduce__(self):
        return (self.__class__, (self._params[0], list(self._params[1:])))

# ----------------------------------------------------------------------------

# TODO: docstrings are now fucked for energy, gradient, etc.

class CPotentialBase(PotentialBase):
    """
    A baseclass for defining gravitational potentials implemented in C.
    """

    def __init__(self, parameters, units, ndim=3, Wrapper=None):
        super(CPotentialBase, self).__init__(parameters, units=units,
                                             ndim=ndim)

        self.c_parameters = np.array([v.value for v in self.parameters.values()])

        if Wrapper is None:
            # magic to set the c_instance attribute based on the name of the class
            wrapper_name = '{}Wrapper'.format(self.__class__.__name__.replace('Potential', ''))

            from .builtin import cybuiltin
            Wrapper = getattr(cybuiltin, wrapper_name)

        self.c_instance = Wrapper(self.G, self.c_parameters)

    def _energy(self, q, t=0.):
        return self.c_instance.energy(q, t=t)

    def _gradient(self, q, t=0.):
        return self.c_instance.gradient(q, t=t)

    def _density(self, q, t=0.):
        return self.c_instance.density(q, t=t)

    def _hessian(self, q, t=0.):
        return self.c_instance.hessian(q, t=t)

    # ----------------------------------------------------------
    # Overwrite the Python potential method to use Cython method
    # TODO: fix this shite
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

        q = self._remove_units_prepare_shape(q)
        orig_shape,q = self._get_c_valid_arr(q)

        try:
            menc = self.c_instance.mass_enclosed(q, self.G, t=t)
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "mass_enclosed function")

        return menc.reshape(orig_shape[1:]) * self.units['mass']
