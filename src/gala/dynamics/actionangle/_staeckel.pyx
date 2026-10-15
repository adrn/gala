# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: language=c++

import numpy as np

from ...potential.potential.cpotential cimport CPotentialWrapper, CPotential

cdef extern from "dynamics/actionangle/staeckel_fudge.h":
    void staeckel_fudge(CPotential *p, double t, const double *w, const double *delta,
                        size_t n, double *out) except + nogil


# TODO: if we do the loop here instead, it could be easier to parallelize the loop over
# phase-space positions?
def staeckel_fudge_c(
    potential, const double[:, ::1] w, const double[::1] delta, double t=0.
):
    """
    Low-level Staeckel fudge: ``w`` is (n, 6) Cartesian phase-space positions and
    ``delta`` is (n,) focal distances, both in the potential's unit system. Returns an
    (n, 9) array of [Jr, Jphi, Jz, theta_r, theta_phi, theta_z, Omega_r, Omega_phi,
    Omega_z].
    """
    cdef size_t n = w.shape[0]
    if w.shape[1] != 6 or delta.shape[0] != n:
        raise ValueError("w must have shape (n, 6) and delta shape (n,)")

    cdef double[:, ::1] out = np.empty((n, 9))
    cdef CPotential* cp = (<CPotentialWrapper>(potential.c_instance)).cpotential
    if n > 0:
        with nogil:
            staeckel_fudge(cp, t, &w[0, 0], &delta[0], n, &out[0, 0])
    return np.asarray(out)
