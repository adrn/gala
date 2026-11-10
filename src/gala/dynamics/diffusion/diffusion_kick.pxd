# Shared declaration of the GSL-free kick hand-off struct (see src/diffusion_kick.h),
# cimported by the diffusion extension and by the leapfrog integrator.

cdef extern from "dynamics/diffusion/src/diffusion_kick.h":
    ctypedef void (*kickfunc)(double t, double dt, double *x, double *v,
                              double *dx, double *dv, void *state, void *rng) noexcept nogil
    ctypedef struct DiffusionKick:
        kickfunc kick
        void *state
        void *rng
