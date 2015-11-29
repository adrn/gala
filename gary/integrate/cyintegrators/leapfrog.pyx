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
from ...potential.cpotential cimport _CPotential

# ctypedef void (*f_type)(int, double*, double*)

cdef void c_init_velocity(_CPotential p, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil:
    cdef int k

    p._gradient(t, x_jm1, grad)

    for k in range(ndim):
        v_jm1_2[k] = v_jm1[k] - grad[k] * dt/2.  # acceleration is minus gradient

cdef void c_leapfrog_step(_CPotential p, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil:
    cdef int k

    # full step the positions
    for k in range(ndim):
        x_jm1[k] = x_jm1[k] + v_jm1_2[k] * dt

    p._gradient(t, x_jm1, grad)  # compute gradient at new position

    # step velocity forward by half step, aligned w/ position, then
    #   finish the full step to leapfrog over position
    for k in range(ndim):
        v_jm1[k] = v_jm1_2[k] - grad[k] * dt/2.
        v_jm1_2[k] = v_jm1_2[k] - grad[k] * dt

cpdef leapfrog_integrate_potential(_CPotential potential, double [:,::1] w0,
                                   double[::1] t):

    cdef:
        # temporary scalars
        int i,j,k
        int ndim = w0.shape[0] // 2
        int n = w0.shape[1]

        int nsteps = len(t)
        double dt = t[1]-t[0]

        # temporary array containers
        double[::1] grad = np.zeros(ndim)
        double[:,::1] v_jm1_2 = np.zeros((ndim,n))

        # return arrays
        double[:,:,::1] all_w = np.zeros((2*ndim,nsteps+1,n))

    # save initial conditions
    all_w[:,0,:] = w0.copy()

    with nogil:
        # first initialize the velocities so they are evolved by a
        #   half step relative to the positions
        for i in range(n):
            c_init_velocity(potential, ndim, t[0], dt,
                            &all_w[0,0,i], &all_w[ndim,0,i], &v_jm1_2[0,i], &grad[0])

        for j in range(1,nsteps+1):
            for i in range(n):
                for k in range(ndim):
                    all_w[k,j,i] = all_w[k,j-1,i]
                    grad[k] = 0.

                c_leapfrog_step(potential, ndim, t[j], dt,
                                &all_w[0,j,i], &all_w[ndim,j,i], &v_jm1_2[0,i], &grad[0])

    return np.asarray(t), np.asarray(all_w)
