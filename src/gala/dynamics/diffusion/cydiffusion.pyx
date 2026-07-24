# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3
# cython: language=c++

"""
Cython layer for the trial SDE / diffusion integrator.

Provides:
  - ``CDiffusionWrapper`` and per-model subclasses that hold a ``CDiffusion``
    struct pointing at a compiled C diffusion model (mirrors the
    ``CPotentialWrapper`` pattern).
  - ``stochastic_leapfrog_integrate``: a fixed-step leapfrog integrator with an
    added Euler-Maruyama stochastic velocity kick each step. The deterministic
    part is a verbatim copy of the leapfrog step used by
    ``gala.integrate.cyintegrators.leapfrog`` (cdef functions there are module
    -static and cannot be shared across extensions, so they are copied here).
"""

import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sqrt

from ...potential.potential.cpotential cimport CPotentialWrapper, CPotential, c_gradient
from ...potential.frame import StaticFrame
from ..._cconfig cimport USE_GSL


# ----------------------------------------------------------------------------
# Deterministic leapfrog step (copied from integrate/cyintegrators/leapfrog.pyx)

cdef void c_init_velocity(CPotential *p, size_t n, int half_ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) noexcept nogil:
    cdef int i, k
    c_gradient(p, n, t, x_jm1, grad)
    for k in range(half_ndim):
        for i in range(n):
            v_jm1_2[i + k * n] = v_jm1[i + k * n] - grad[i + k * n] * dt / 2.


cdef void c_leapfrog_step(CPotential *p, size_t n, int half_ndim, double t, double dt,
                          double *x_jm1, double *v_jm1, double *v_jm1_2, double *grad) noexcept nogil:
    cdef int i, k

    # full step the positions
    for k in range(half_ndim):
        for i in range(n):
            x_jm1[i + k * n] = x_jm1[i + k * n] + v_jm1_2[i + k * n] * dt

    c_gradient(p, n, t, x_jm1, grad)  # gradient at new position

    # synchronized velocity (v_jm1) and half-step-ahead velocity (v_jm1_2)
    for k in range(half_ndim):
        for i in range(n):
            v_jm1[i + k * n] = v_jm1_2[i + k * n] - grad[i + k * n] * dt / 2.
            v_jm1_2[i + k * n] = v_jm1_2[i + k * n] - grad[i + k * n] * dt


# ----------------------------------------------------------------------------
# Stochastic velocity kick (Euler-Maruyama) for a single orbit

cdef void c_apply_kick(CDiffusion *diff, void *rng, double t, int half_ndim, size_t n,
                       size_t i, double *x_ptr, double *v_sync, double *v_stag,
                       double dt, double sqrt_dt, double *q_i, double *v_i,
                       double *drift, double *M, double *L, double *xi) noexcept nogil:
    cdef int k, l
    cdef double dv

    # gather this orbit's position / synchronized velocity
    for k in range(half_ndim):
        q_i[k] = x_ptr[i + k * n]
        v_i[k] = v_sync[i + k * n]

    # evaluate the diffusion model at (t, q, v)
    diff.func(t, diff.parameters, q_i, v_i, half_ndim, drift, M, diff.state)

    # obtain the noise factor L (L L^T = D)
    if diff.returns_factor == 0:
        gala_diffusion_cholesky(M, half_ndim, L)
    else:
        for k in range(half_ndim * half_ndim):
            L[k] = M[k]

    # draw standard-normal noise
    for k in range(half_ndim):
        xi[k] = gala_diffusion_rng_gaussian(rng)

    # dv = drift * dt + L @ (sqrt(dt) * xi), applied to both velocity copies
    for k in range(half_ndim):
        dv = drift[k] * dt
        for l in range(half_ndim):
            dv = dv + L[k * half_ndim + l] * sqrt_dt * xi[l]
        v_sync[i + k * n] = v_sync[i + k * n] + dv
        v_stag[i + k * n] = v_stag[i + k * n] + dv


# ----------------------------------------------------------------------------
# Diffusion model wrappers

cdef class CDiffusionWrapper:
    """
    Cython wrapper holding a ``CDiffusion`` struct. Subclasses assign the C model
    function pointer (and the ``returns_factor`` flag) in ``__init__``.
    """

    cpdef init(self, double[::1] parameters, int n_dim, int returns_factor):
        self._params = np.ascontiguousarray(parameters, dtype=np.float64)
        self.diffusion.n_dim = n_dim
        self.diffusion.returns_factor = returns_factor
        self.diffusion.n_params = self._params.shape[0]
        if self._params.shape[0] > 0:
            self.diffusion.parameters = &self._params[0]
        else:
            self.diffusion.parameters = NULL
        self.diffusion.func = NULL
        self.diffusion.state = NULL

    cdef CDiffusion* get_ptr(self) noexcept nogil:
        return &self.diffusion


cdef class ConstantDiagDiffusionWrapper(CDiffusionWrapper):
    def __init__(self, parameters, n_dim):
        self.init(np.ascontiguousarray(parameters, dtype=np.float64), n_dim, 1)
        self.diffusion.func = <diffusionfunc>constant_diag_diffusion


cdef class ConstantTensorDiffusionWrapper(CDiffusionWrapper):
    def __init__(self, parameters, n_dim):
        self.init(np.ascontiguousarray(parameters, dtype=np.float64), n_dim, 0)
        self.diffusion.func = <diffusionfunc>constant_tensor_diffusion


cdef class ExampleRadialDiffusionWrapper(CDiffusionWrapper):
    def __init__(self, parameters, n_dim):
        self.init(np.ascontiguousarray(parameters, dtype=np.float64), n_dim, 1)
        self.diffusion.func = <diffusionfunc>example_radial_diffusion


# ----------------------------------------------------------------------------
# The integrator

cpdef stochastic_leapfrog_integrate(hamiltonian, double[:, ::1] w0, double[::1] t,
                                    diffusion, unsigned long long seed,
                                    int save_all=1):
    """
    w0: shape (ndim, n)  [ndim = full phase-space dimension = 2 * half_ndim]
    returns: (t, w) with w shape (ndim, [ntimes,] n)
    """
    if USE_GSL != 1:
        raise RuntimeError(
            "The diffusion / SDE integrator requires GSL support. Please install "
            "GSL and rebuild gala with GSL support to use this functionality."
        )

    if not hamiltonian.c_enabled:
        raise TypeError("Input Hamiltonian object does not support C-level access.")

    if not isinstance(hamiltonian.frame, StaticFrame):
        raise TypeError(
            "The diffusion integrator is currently only supported for StaticFrame, "
            f"not {hamiltonian.frame.__class__.__name__}"
        )

    cdef:
        size_t i
        int j, k
        int ndim = w0.shape[0]
        size_t n = w0.shape[1]
        int half_ndim = ndim // 2

        int ntimes = len(t)
        double dt = t[1] - t[0]
        double sqrt_dt = sqrt(dt)

        double[:, ::1] grad_v = np.zeros((half_ndim, n))
        double[:, ::1] v_jm1_2 = np.zeros((half_ndim, n))

        # per-orbit scratch buffers
        double[::1] q_i = np.zeros(half_ndim)
        double[::1] v_i = np.zeros(half_ndim)
        double[::1] drift = np.zeros(half_ndim)
        double[::1] M = np.zeros(half_ndim * half_ndim)
        double[::1] L = np.zeros(half_ndim * half_ndim)
        double[::1] xi = np.zeros(half_ndim)

        double[:, :, ::1] all_w
        double[:, ::1] tmp_w

        CPotential* cp = (<CPotentialWrapper>(hamiltonian.potential.c_instance)).cpotential
        CDiffusion* diff = (<CDiffusionWrapper>(diffusion.c_instance)).get_ptr()
        void* rng

    if diff.n_dim != half_ndim:
        raise ValueError(
            f"Diffusion model dimensionality ({diff.n_dim}) does not match the "
            f"phase-space dimensionality of the initial conditions ({half_ndim})."
        )

    if save_all:
        all_w = np.empty((ndim, ntimes, n))
        all_w[:, 0, :] = w0

    tmp_w = w0.copy()

    rng = gala_diffusion_rng_alloc(seed)
    try:
        with nogil:
            # prime velocities a half-step behind the positions
            c_init_velocity(cp, n, half_ndim, t[0], dt,
                            &tmp_w[0, 0], &tmp_w[half_ndim, 0],
                            &v_jm1_2[0, 0], &grad_v[0, 0])

            for j in range(1, ntimes, 1):
                grad_v[:] = 0.

                # deterministic leapfrog drift
                c_leapfrog_step(cp, n, half_ndim, t[j], dt,
                                &tmp_w[0, 0], &tmp_w[half_ndim, 0],
                                &v_jm1_2[0, 0], &grad_v[0, 0])

                # stochastic velocity kick per orbit
                for i in range(n):
                    c_apply_kick(diff, rng, t[j], half_ndim, n, i,
                                 &tmp_w[0, 0], &tmp_w[half_ndim, 0], &v_jm1_2[0, 0],
                                 dt, sqrt_dt,
                                 &q_i[0], &v_i[0], &drift[0], &M[0], &L[0], &xi[0])

                if save_all:
                    for k in range(ndim):
                        for i in range(n):
                            all_w[k, j, i] = tmp_w[k, i]
    finally:
        gala_diffusion_rng_free(rng)

    if save_all:
        return np.asarray(t), np.asarray(all_w)
    return np.asarray(t)[ntimes - 1:], np.asarray(tmp_w)
