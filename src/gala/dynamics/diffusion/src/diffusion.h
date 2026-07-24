/*
    C-level scaffolding for velocity-space diffusion models used by the trial
    SDE / diffusion integrator (see gala.dynamics.diffusion).

    A "diffusion model" is a compiled C function that, given a time, position,
    and velocity, fills in a (velocity) drift vector and an n_dim x n_dim matrix
    describing the local diffusion. This mirrors the function-pointer + parameter
    array pattern used by gala's builtin C potentials (see funcdefs.h /
    cpotential.h), but for diffusion coefficients rather than forces.
*/
#ifndef _GALA_DIFFUSION_H
#define _GALA_DIFFUSION_H

#include <stddef.h>

/*
    Diffusion coefficient function pointer type.

    Arguments:
        t       - time
        pars    - parameter array (pars[0]... ; NOTE: no G, unlike potentials)
        q       - position, length n_dim
        v       - velocity, length n_dim
        n_dim   - number of spatial dimensions (typically 3)
        drift   - OUTPUT: deterministic velocity drift, length n_dim. Fill with
                  0 for pure diffusion; use for a noise-induced / Fokker-Planck
                  drift-correction term.
        M       - OUTPUT: n_dim x n_dim matrix (row-major). Interpreted as either
                  the diffusion tensor D or a pre-computed factor B depending on
                  the CDiffusion.returns_factor flag (see below).
        state   - opaque per-model state (may be NULL).
*/
typedef void (*diffusionfunc)(double t, double *pars, double *q, double *v,
                              int n_dim, double *drift, double *M, void *state);

typedef struct _CDiffusion CDiffusion;
struct _CDiffusion {
    int n_dim;          // spatial dimensionality (velocity-space dimension)
    int returns_factor; // 0 => M is the diffusion tensor D (needs Cholesky)
                        // 1 => M is a factor B with B B^T = D (applied directly)
    int n_params;       // number of entries in `parameters`
    diffusionfunc func; // the model function
    double *parameters; // parameter array (owned by the Cython wrapper)
    void *state;        // opaque model state (may be NULL)
};

#ifdef __cplusplus
extern "C" {
#endif

/*
    Random number generation. Backed by GSL (gsl_rng / gsl_ran_ugaussian) when
    gala is compiled with GSL; otherwise these are no-ops and the Python layer
    refuses to run. The RNG pointer is opaque to the caller.
*/
void *gala_diffusion_rng_alloc(unsigned long long seed);
double gala_diffusion_rng_gaussian(void *rng); /* draw ~ N(0, 1) */
void gala_diffusion_rng_free(void *rng);

/*
    Lower-triangular Cholesky factorization of a symmetric n_dim x n_dim matrix
    M (row-major) into L (row-major, lower triangular) so that L L^T = M. Degenerate
    / non-positive-definite directions are handled by clamping to zero variance,
    so a positive-semidefinite (including zero) M produces a valid best-effort
    factor. Returns 0 (reserved for future error signaling).
*/
int gala_diffusion_cholesky(const double *M, int n_dim, double *L);

/* ---- builtin diffusion models ---- */

/*
    Constant, diagonal velocity diffusion. Declared with returns_factor=1:
        pars = [D_0, ..., D_{n_dim-1}]  (per-component diffusion rates)
    fills drift=0 and B = diag(sqrt(D_i)), so Cov(kick) = diag(D_i) * dt.
*/
void constant_diag_diffusion(double t, double *pars, double *q, double *v,
                             int n_dim, double *drift, double *M, void *state);

/*
    Constant, full-tensor velocity diffusion. Declared with returns_factor=0:
        pars = flattened n_dim x n_dim symmetric diffusion tensor D (row-major)
    fills drift=0 and M = D; the integrator Choleskys it. Cov(kick) = D * dt.
*/
void constant_tensor_diffusion(double t, double *pars, double *q, double *v,
                               int n_dim, double *drift, double *M, void *state);

/*
    Example position-dependent diagonal diffusion (a TEMPLATE demonstrating how a
    model can depend on q/v). Declared with returns_factor=1:
        pars = [D_0, ..., D_{n_dim-1}, r_s]
    fills B = diag(sqrt(D_i)) * exp(-|q| / r_s). Replace with real ISM / impulsive
    -kick prescriptions following this same shape.
*/
void example_radial_diffusion(double t, double *pars, double *q, double *v,
                              int n_dim, double *drift, double *M, void *state);

#ifdef __cplusplus
}
#endif

#endif
