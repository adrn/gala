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

from cython.parallel import prange

from ..potential.cpotential cimport _CPotential

cdef void c_init_velocity(_CPotential p, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil:
    cdef int k

    p._gradient(x_jm1, grad)

    for k in range(ndim):
        v_jm1_2[k] = v_jm1[k] - grad[k] * dt/2.  # acceleration is minus gradient

cdef void c_leapfrog_step(_CPotential p, int ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil:
    cdef int k

    # full step the positions
    for k in range(ndim):
        x_jm1[k] = x_jm1[k] + v_jm1_2[k] * dt

    p._gradient(x_jm1, grad)  # compute gradient at new position

    # step velocity forward by half step, aligned w/ position, then
    #   finish the full step to leapfrog over position
    for k in range(ndim):
        v_jm1[k] = v_jm1_2[k] - grad[k] * dt/2.
        v_jm1_2[k] = v_jm1_2[k] - grad[k] * dt

cpdef cy_leapfrog_run(_CPotential potential, double [:,::1] w0,
                      double dt, int nsteps, double t1):
    cdef int i,j,k
    cdef int n = w0.shape[0]
    cdef int ndim = w0.shape[1] // 2
    cdef double t

    cdef double[::1] grad = np.zeros(ndim)
    cdef double[::1] all_t = np.zeros(nsteps+1)
    cdef double[:,:,::1] all_w = np.zeros((nsteps+1,n,2*ndim))
    cdef double[:,::1] v_jm1_2 = np.zeros((n,ndim))

    # save initial conditions
    all_w[0,:,:] = w0.copy()

    with nogil:
        # first initialize the velocities so they are evolved by a
        #   half step relative to the positions
        for i in range(n):
            c_init_velocity(potential, ndim, t1, dt,
                            &all_w[0,i,0], &all_w[0,i,ndim], &v_jm1_2[i,0], &grad[0])

        t = t1  # initial time
        all_t[0] = t
        for j in range(1,nsteps+1):
            t += dt
            all_t[j] = t
            for i in range(n):
                for k in range(ndim):
                    all_w[j,i,k] = all_w[j-1,i,k]
                    grad[k] = 0.

                c_leapfrog_step(potential, ndim, t, dt,
                                &all_w[j,i,0], &all_w[j,i,ndim], &v_jm1_2[i,0], &grad[0])

    return np.array(all_t), np.array(all_w)
