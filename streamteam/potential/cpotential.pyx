# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

# Project
from .core import Potential

class CPotential(Potential):
    """
    A baseclass for representing gravitational potentials. You must specify
    a function that evaluates the potential value (func). You may also
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

    def __init__(self, c_class, parameters=dict()):
        # store C instance
        self.c_instance = c_class(**parameters)

        # HACK?
        super(CPotential, self).__init__(func=lambda x: x, parameters=parameters)

        # self.value = getattr(self.c_instance, 'value')
        # self.gradient = getattr(self.c_instance, 'gradient', None)
        # self.hessian = getattr(self.c_instance, 'hessian', None)

    def value(self, q):
        """
        Compute the value of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the value of the potential.
        """
        return self.c_instance.value(np.array(q))

    def gradient(self, q):
        """
        Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the gradient.
        """
        try:
            return self.c_instance.gradient(np.array(q))
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "gradient function")

    def hessian(self, q):
        """
        Compute the Hessian of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the Hessian.
        """
        try:
            return self.c_instance.hessian(np.array(q))
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "Hessian function")

    # ----------------------------
    # Functions of the derivatives
    # ----------------------------
    def acceleration(self, q):
        """
        Compute the acceleration due to the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the acceleration.
        """
        try:
            return -self.c_instance.gradient(np.array(q))
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "gradient function")

# ==============================================================================

cdef class _CPotential:

    cpdef value(self, double[:,::1] q):
        cdef int nparticles, ndim
        nparticles = q.shape[0]
        ndim = q.shape[1]

        cdef double [::1] pot = np.empty(nparticles)
        self._value(q, pot, nparticles)
        return np.array(pot)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public void _value(self, double[:,::1] q, double[::1] pot, int nparticles) nogil:
        for i in range(nparticles):
            pot[i] = 0.

    # -------------------------------------------------------------
    cpdef gradient(self, double[:,::1] q):
        cdef int nparticles, ndim
        nparticles = q.shape[0]
        ndim = q.shape[1]

        cdef double [:,::1] grad = np.empty((nparticles,ndim))
        self._gradient(q, grad, nparticles)
        return np.array(grad)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public void _gradient(self, double[:,::1] r, double[:,::1] grad, int nparticles) nogil:
        for i in range(nparticles):
            grad[i,0] = 0.
            grad[i,1] = 0.
            grad[i,2] = 0.

    # -------------------------------------------------------------
    cpdef hessian(self, double[:,::1] w):
        cdef int nparticles, ndim
        nparticles = w.shape[0]
        ndim = w.shape[1]

        cdef double [:,::1] hess = np.empty((nparticles,ndim,ndim))
        self._hessian(w, hess, nparticles)
        return np.array(hess)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef public void _hessian(self, double[:,::1] w, double[:,::1] acc, int nparticles) nogil:
        for i in range(nparticles):
            acc[i,0] = 0.
            acc[i,1] = 0.
            acc[i,2] = 0.

    # -------------------------------------------------------------
    cpdef mass_enclosed(self, double[:,::1] q):
        cdef int nparticles, ndim
        nparticles = q.shape[0]
        ndim = q.shape[1]

        cdef double [:,::1] mass = np.empty((nparticles,))
        self._mass_enclosed(q, mass, nparticles)
        return np.array(mass)
