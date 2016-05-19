# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Leapfrog integration in Cython. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

# Project
from ...potential.cpotential cimport CPotentialWrapper

cdef extern from "src/cpotential.h":
    ctypedef struct CPotential:
        pass
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil

cdef void c_init_velocity(CPotential *p, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil:
    cdef int k

    c_gradient(p, t, x_jm1, grad)

    for k in range(ndim):
        v_jm1_2[k] = v_jm1[k] - grad[k] * dt/2.  # acceleration is minus gradient

cdef void c_leapfrog_step(CPotential *p, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil:
    cdef int k

    # full step the positions
    for k in range(ndim):
        x_jm1[k] = x_jm1[k] + v_jm1_2[k] * dt

    c_gradient(p, t, x_jm1, grad)  # compute gradient at new position

    # step velocity forward by half step, aligned w/ position, then
    #   finish the full step to leapfrog over position
    for k in range(ndim):
        v_jm1[k] = v_jm1_2[k] - grad[k] * dt/2.
        v_jm1_2[k] = v_jm1_2[k] - grad[k] * dt

cpdef leapfrog_integrate_potential(CPotentialWrapper p, double [:,::1] w0,
                                   double[::1] t):
    """
    CAUTION: Interpretation of axes is different here! We need the
    arrays to be C ordered and easy to iterate over, so here the
    axes are (norbits, ndim).
    """
    cdef:
        # temporary scalars
        int i,j,k
        int n = w0.shape[0]
        int ndim = w0.shape[1] // 2

        int ntimes = len(t)
        double dt = t[1]-t[0]

        # temporary array containers
        double[::1] grad = np.zeros(ndim)
        double[:,::1] v_jm1_2 = np.zeros((n,ndim))

        # return arrays
        double[:,:,::1] all_w = np.zeros((ntimes,n,2*ndim))

    # save initial conditions
    all_w[0,:,:] = w0.copy()

    with nogil:
        # first initialize the velocities so they are evolved by a
        #   half step relative to the positions
        for i in range(n):
            c_init_velocity(&(p.cpotential), ndim, t[0], dt,
                            &all_w[0,i,0], &all_w[0,i,ndim], &v_jm1_2[i,0], &grad[0])

        for j in range(1,ntimes,1):
            for i in range(n):
                for k in range(ndim):
                    all_w[j,i,k] = all_w[j-1,i,k]
                    grad[k] = 0.

                c_leapfrog_step(&(p.cpotential), ndim, t[j], dt,
                                &all_w[j,i,0], &all_w[j,i,ndim], &v_jm1_2[i,0], &grad[0])

    return np.asarray(t), np.asarray(all_w)
