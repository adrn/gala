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

    def value(self, q):
        """
        value(q)

        Compute the value of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the value of the potential.
        """
        return self.c_instance.value(np.atleast_2d(q).copy())

    def gradient(self, q):
        """
        gradient(q)

        Compute the gradient of the potential at the given position(s).

        Parameters
        ----------
        q : array_like, numeric
            Position to compute the gradient.
        """
        try:
            return self.c_instance.gradient(np.atleast_2d(q).copy())
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "gradient function")

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
            return self.c_instance.hessian(np.atleast_2d(q).copy())
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "Hessian function")

    # ----------------------------
    # Functions of the derivatives
    # ----------------------------
    def mass_enclosed(self, q):
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
            return self.c_instance.mass_enclosed(np.atleast_2d(q).copy())
        except AttributeError,TypeError:
            raise ValueError("Potential C instance has no defined "
                             "mass_enclosed function")

# ==============================================================================

cdef class _CPotential:

    def __setstate__(self, d):
        # for k,v in d.items():
        #     setattr(self, k, v)
        pass

    def __getstate__(self):
        return None

    cpdef value(self, double[:,::1] q):
        cdef int nparticles, ndim, k
        nparticles = q.shape[0]
        ndim = q.shape[1]

        cdef double [::1] pot = np.zeros((nparticles,))
        for k in range(nparticles):
            pot[k] = self._value(&q[k,0])

        return np.array(pot)

    cdef public double _value(self, double* q) nogil:
        return 0.

    # -------------------------------------------------------------
    cpdef gradient(self, double[:,::1] q):
        cdef int nparticles, ndim, k
        nparticles = q.shape[0]
        ndim = q.shape[1]

        cdef double [:,::1] grad = np.zeros((nparticles,ndim))
        for k in range(nparticles):
            self._gradient(&q[k,0], &grad[k,0])

        return np.array(grad)

    cdef public void _gradient(self, double *r, double *grad) nogil:
        grad[0] = 0.
        grad[1] = 0.
        grad[2] = 0.

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
    cpdef mass_enclosed(self, double[:,::1] q):
        cdef int nparticles, k
        nparticles = q.shape[0]

        cdef double [::1] epsilon = np.zeros(3)
        cdef double [::1] mass = np.zeros((nparticles,))
        for k in range(nparticles):
            mass[k] = self._mass_enclosed(&q[k,0], &epsilon[0], self.G)
        return np.array(mass)

    cdef public double _mass_enclosed(self, double *q, double *epsilon, double Gee) nogil:
        cdef double h, r, dPhi_dr

        # Fractional step-size
        h = 0.01

        # Step-size for estimating radial gradient of the potential
        r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])

        for j in range(3):
            epsilon[j] = h * q[j]/r + q[j]
        dPhi_dr = self._value(epsilon)

        for j in range(3):
            epsilon[j] = h * q[j]/r - q[j]
        dPhi_dr -= self._value(epsilon)

        return fabs(r*r * dPhi_dr / Gee / (2.*h))

# ==============================================================================

class CCompositePotential(CPotentialBase):
    """

    TODO!

    A baseclass for representing gravitational potentials. You must specify
    a function that evaluates the potential value (func). You may also
    optionally add a function that computes derivatives (gradient), and a
    function to compute second derivatives (the Hessian) of the potential.

    Parameters
    ----------
    TODO
    """

    def __init__(self, **kwargs):
        # hurm?
        self.c_instance = _CCompositePotential([p.c_instance for p in kwargs.values()])
        super(CPotentialBase, self).__init__(func=lambda x: x, parameters=dict())

from cpython cimport PyObject
cdef class _CCompositePotential(_CPotential):

    cdef public list py_instances
    cdef public int ninstances
    cdef PyObject *obj_list[100]
    cdef public double G

    def __init__(self, instance_list):
        """ Need a list of instances of _CPotential classes """
        self.py_instances = list(instance_list)
        self.ninstances = len(instance_list)
        self._more_init()

    cpdef _more_init(self):
        for i in range(self.ninstances):
            self.obj_list[i] = <PyObject *>self.py_instances[i]

    cdef public double _value(self, double* q) nogil:
        # whoa this is some whack cython wizardry right here
        #   (stolen from stackoverflow)
        cdef double v = 0.
        for i in range(self.ninstances):
            v += (<_CPotential>(self.obj_list[i]))._value(q)
        return v

    cdef public void _gradient(self, double *r, double *grad) nogil:
        # whoa this is some whack cython wizardry right here
        #   (stolen from stackoverflow)
        cdef double v = 0.

        for i in range(self.ninstances):
            (<_CPotential>(self.obj_list[i]))._gradient(r, grad)

    cdef public void _hessian(self, double *w, double *hess) nogil:
        # whoa this is some whack cython wizardry right here
        #   (stolen from stackoverflow)
        for i in range(self.ninstances):
            (<_CPotential>(self.obj_list[i]))._hessian(w, hess)

    cdef public double _mass_enclosed(self, double *q, double *epsilon, double Gee) nogil:
        cdef double mm = 0.
        for i in range(self.ninstances):
            mm += (<_CPotential>(self.obj_list[i]))._mass_enclosed(q, epsilon, Gee)
        return mm
