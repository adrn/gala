# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

""" Leapfrog integration in Cython. """

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

# Project
from ...potential.potential.cpotential cimport CPotentialWrapper
from ...potential.frame import StaticFrame

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrameType:
        pass

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        pass

cdef extern from "potential/src/cpotential.h":
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil

cdef void c_init_velocity(CPotential *p, int half_ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil:
    cdef int k

    c_gradient(p, t, x_jm1, grad)

    for k in range(half_ndim):
        v_jm1_2[k] = v_jm1[k] - grad[k] * dt/2.  # acceleration is minus gradient

cdef void c_leapfrog_step(CPotential *p, int half_ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) nogil:
    cdef int k

    # full step the positions
    for k in range(half_ndim):
        x_jm1[k] = x_jm1[k] + v_jm1_2[k] * dt

    c_gradient(p, t, x_jm1, grad)  # compute gradient at new position

    # step velocity forward by half step, aligned w/ position, then
    #   finish the full step to leapfrog over position
    for k in range(half_ndim):
        v_jm1[k] = v_jm1_2[k] - grad[k] * dt/2.
        v_jm1_2[k] = v_jm1_2[k] - grad[k] * dt

cpdef leapfrog_integrate_hamiltonian(hamiltonian, double [:, ::1] w0, double[::1] t,
                                     int store_all=1):
    """
    CAUTION: Interpretation of axes is different here! We need the
    arrays to be C ordered and easy to iterate over, so here the
    axes are (norbits, ndim).
    """

    if not hamiltonian.c_enabled:
        raise TypeError("Input Hamiltonian object does not support C-level access.")

    if not isinstance(hamiltonian.frame, StaticFrame):
        raise TypeError(
            "Leapfrog integration is currently only supported for StaticFrame, "
            f"not {hamiltonian.frame.__class__.__name__}"
        )

    cdef:
        # temporary scalars
        int i, j, k
        int n = w0.shape[0]
        int ndim = w0.shape[1]
        int half_ndim = ndim // 2

        int ntimes = len(t)
        double dt = t[1]-t[0]

        # temporary array containers
        double[::1] grad = np.zeros(half_ndim)
        double[:, ::1] v_jm1_2 = np.zeros((n, half_ndim))

        # return arrays
        double[:, :, ::1] all_w
        double[:, ::1] tmp_w = np.zeros((n, ndim))

        # whoa, so many dots
        CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential

    if store_all:
        all_w = np.zeros((ntimes, n, ndim))

        # save initial conditions
        all_w[0, :, :] = w0.copy()

    tmp_w = w0.copy()

    with nogil:
        # first initialize the velocities so they are evolved by a
        #   half step relative to the positions
        for i in range(n):
            c_init_velocity(&cp, half_ndim, t[0], dt,
                            &tmp_w[i, 0], &tmp_w[i, half_ndim],
                            &v_jm1_2[i, 0], &grad[0])

        for j in range(1, ntimes, 1):
            for i in range(n):
                for k in range(half_ndim):
                    grad[k] = 0.

                c_leapfrog_step(&cp, half_ndim, t[j], dt,
                                &tmp_w[i, 0], &tmp_w[i, half_ndim],
                                &v_jm1_2[i, 0],
                                &grad[0])

                if store_all:
                    for k in range(ndim):
                        all_w[j, i, k] = tmp_w[i, k]

    if store_all:
        return np.asarray(t), np.asarray(all_w)
    else:
        return np.asarray(t[-1:]), np.asarray(tmp_w)
