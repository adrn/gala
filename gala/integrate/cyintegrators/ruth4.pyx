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
from ...potential import NullPotential

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrameType:
        pass

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        pass

cdef extern from "potential/src/cpotential.h":
    void c_gradient(CPotential *p, double t, double *q, double *grad) nogil
    void c_nbody_gradient_symplectic(
        CPotential **pots, double t, double *q,
        double *nbody_q, int nbody, int nbody_i,
        int ndim, double *grad
    ) nogil


cdef void c_ruth4_step(CPotential *p, int half_ndim, double t, double dt,
                       double *cs, double *ds,
                       double *w, double *grad) nogil:
    cdef:
        int j, k

    for j in range(4):
        for k in range(half_ndim):
             grad[k] = 0.
        c_gradient(p, t, w, grad)
        for k in range(half_ndim):
            w[half_ndim + k] = w[half_ndim + k] - ds[j] * grad[k] * dt
            w[k] = w[k] + cs[j] * w[half_ndim + k] * dt

cpdef ruth4_integrate_hamiltonian(hamiltonian,
                                  double[:, ::1] w0,
                                  double[::1] t,
                                  int store_all=1):
    """
    CAUTION: Interpretation of axes is different here! We need the
    arrays to be C ordered and easy to iterate over, so here the
    axes are (norbits, ndim).
    """

    if not hamiltonian.c_enabled:
        raise TypeError("Input Hamiltonian object does not support C-level access.")

    if not isinstance(hamiltonian.frame, StaticFrame):
        raise TypeError("Leapfrog integration is currently only supported "
                        "for StaticFrame, not {}."
                        .format(hamiltonian.frame.__class__.__name__))

    cdef:
        # temporary scalars
        int i, j, k
        int n = w0.shape[0]
        int ndim = w0.shape[1]
        int half_ndim = ndim // 2

        int ntimes = len(t)
        double dt = t[1] - t[0]

        # Integrator coefficients
        double two_13 = 2 ** (1./3.)
        double[::1] cs = np.array([
            1. / (2. * (2. - two_13)),
            (1. - two_13) / (2.*(2. - two_13)),
            (1. - two_13) / (2.*(2. - two_13)),
            1. / (2.*(2. - two_13))
        ], dtype='f8')

        double[::1] ds = np.array([
            0.,
            1. / (2. - two_13),
            -two_13 / (2. - two_13),
            1. / (2. - two_13)
        ], dtype='f8')

        # temporary array containers
        double[::1] grad = np.zeros(half_ndim)

        # return arrays
        double[:, :, ::1] all_w
        double[:, ::1] tmp_w

        # whoa, so many dots
        CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential

    if store_all:
        all_w = np.zeros((ntimes, n, ndim))

        # save initial conditions
        all_w[0, :, :] = w0.copy()

    tmp_w = w0.copy()

    with nogil:

        for j in range(1, ntimes, 1):
            for i in range(n):
                c_ruth4_step(&cp, half_ndim, t[j], dt,
                             &cs[0], &ds[0],
                             &tmp_w[i, 0], &grad[0])

                if store_all:
                    for k in range(ndim):
                        all_w[j, i, k] = tmp_w[i, k]

    if store_all:
        return np.asarray(t), np.asarray(all_w)
    else:
        return np.asarray(t[-1:]), np.asarray(tmp_w)


# -------------------------------------------------------------------------------------
# N-body stuff - TODO: to be moved, because this is a HACK!

cdef void c_ruth4_step_nbody(
    CPotential *p, int half_ndim, double t, double dt,
    CPotential **pots, double *w_nbody, int nbody, int nbody_i,
    double *cs, double *ds,
    double *w, double *grad
) nogil:
    cdef:
        int j, k

    for j in range(4):
        for k in range(half_ndim):
             grad[k] = 0.

        c_gradient(p, t, w, grad)
        c_nbody_gradient_symplectic(pots, t, w, w_nbody, nbody, nbody_i, half_ndim, grad)

        for k in range(half_ndim):
            w[half_ndim + k] = w[half_ndim + k] - ds[j] * grad[k] * dt
            w[k] = w[k] + cs[j] * w[half_ndim + k] * dt

cpdef ruth4_integrate_nbody(hamiltonian, double [:, ::1] w0, double[::1] t,
                            list particle_potentials, int store_all=1):
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

         # Integrator coefficients
        double two_13 = 2 ** (1./3.)
        double[::1] cs = np.array([
            1. / (2. * (2. - two_13)),
            (1. - two_13) / (2.*(2. - two_13)),
            (1. - two_13) / (2.*(2. - two_13)),
            1. / (2.*(2. - two_13))
        ], dtype='f8')

        double[::1] ds = np.array([
            0.,
            1. / (2. - two_13),
            -two_13 / (2. - two_13),
            1. / (2. - two_13)
        ], dtype='f8')

        # temporary array containers
        double[::1] grad = np.zeros(half_ndim)
        double[:, ::1] v_jm1_2 = np.zeros((n, half_ndim))

        # return arrays
        double[:, :, ::1] all_w
        double[:, ::1] tmp_w = np.zeros((n, ndim))

        # whoa, so many dots
        CPotential cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
        CPotential *c_particle_potentials[MAX_NBODY]
        unsigned nbody = 0

    if store_all:
        all_w = np.zeros((ntimes, n, ndim))

        # save initial conditions
        all_w[0, :, :] = w0.copy()

    for pot in particle_potentials:
        if not isinstance(pot, NullPotential):
            nbody += 1

    # Extract the CPotential objects from the particle potentials.
    for i in range(n):
        c_particle_potentials[i] = &(<CPotentialWrapper>(particle_potentials[i].c_instance)).cpotential

    tmp_w = w0.copy()

    with nogil:

        for j in range(1, ntimes, 1):
            for i in range(n):
                c_ruth4_step_nbody(
                    &cp, half_ndim, t[j], dt,
                    &c_particle_potentials[0], &tmp_w[0, 0], nbody, i,
                    &cs[0], &ds[0],
                    &tmp_w[i, 0], &grad[0]
                )

                if store_all:
                    for k in range(ndim):
                        all_w[j, i, k] = tmp_w[i, k]

    if store_all:
        return np.asarray(t), np.asarray(all_w)
    else:
        return np.asarray(t[-1:]), np.asarray(tmp_w)
