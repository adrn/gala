/*
    C-level scaffolding for velocity/position-space diffusion models used by the trial
    SDE integrator (see gala.dynamics.diffusion). A "diffusion model" fills a drift
    vector and a diffusion tensor (in a chosen basis) at a phase-space point; the kick
    rotates these to Cartesian, factorizes, draws noise, and returns the phase-space
    increment.

    This is 3D-only (n_dim = 3, phase space = 6), matching gala's builtin potentials.
*/
#ifndef _GALA_DIFFUSION_H
#define _GALA_DIFFUSION_H

#include "diffusion_kick.h"

#define GALA_DIFF_NDIM 3
#define GALA_DIFF_PS 6 /* phase-space dimension = 2 * NDIM */

/* Coefficient basis (orientation of the drift/diffusion coefficients). */
#define GALA_DIFF_BASIS_CARTESIAN 0
#define GALA_DIFF_BASIS_CYLINDRICAL 1
/* future: GALA_DIFF_BASIS_VELOCITY_ALIGNED, ..._SPHERICAL */

/*
    Fill drift `mu` (length 6) and matrix `M` (6x6 row-major) at (t, x, v), in the
    model's basis. x, v are Cartesian length-3. `M` is the diffusion tensor D
    (returns_factor == 0) or a pre-computed factor B with B B^T = D (returns_factor==1).
*/
typedef void (*diffusionfunc)(double t, double *pars, double *x, double *v,
                              double *mu, double *M, void *state);

typedef struct _CDiffusion CDiffusion;
struct _CDiffusion {
    int basis;          // one of GALA_DIFF_BASIS_*
    int returns_factor; // 0 => M is tensor D (Cholesky in the kick); 1 => M is factor B
    int n_params;
    diffusionfunc func;
    double *parameters; // owned by the Cython wrapper
    void *state;        // model state (e.g. grid2d_state*); may be NULL
};

#ifdef __cplusplus
extern "C" {
#endif

/* ---- RNG (GSL-backed when USE_GSL, else no-ops) ---- */
void *gala_diffusion_rng_alloc(unsigned long long seed);
void gala_diffusion_rng_set(void *rng, unsigned long long seed);
double gala_diffusion_rng_gaussian(void *rng); /* ~ N(0, 1) */
void gala_diffusion_rng_free(void *rng);

/* Lower-triangular Cholesky of symmetric n x n M (row-major) into L; non-PD directions
   clamped to zero so PSD/zero M is safe. */
int gala_diffusion_cholesky(const double *M, int n, double *L);

/* Build the 3x3 position and velocity rotation blocks (row-major) that map the model's
   basis to Cartesian, given the basis id and the orbit's (x, v). */
void gala_diffusion_build_rotation(int basis, const double *x, const double *v,
                                   double *Qpos, double *Qvel);

/* The kick: DiffusionKick.kick points here. state is a CDiffusion*, rng a gsl_rng*. */
void diffusion_kick_increments(double t, double dt, double *x, double *v,
                               double *dx, double *dv, void *state, void *rng);

/* ---- builtin model functions ---- */
/* Constant: pars = [mu(6), Dflat(36)]; fills mu and M = D directly. */
void constant_diffusion(double t, double *pars, double *x, double *v,
                        double *mu, double *M, void *state);

/* Gridded over cylindrical (R, |z|): state is a grid2d_state*. */
void gridded_cyl_diffusion(double t, double *pars, double *x, double *v,
                           double *mu, double *M, void *state);

/* ---- gridded state management ---- */
/*
    Build a grid2d_state from a regular (R, |z|) grid. `fields` is a contiguous array of
    shape [GALA_DIFF_NFIELDS, nR, nz] (row-major) giving the value of each coefficient
    field at each node. Field order: 0..5 = mu; 6..26 = upper-triangular of the 6x6
    tensor in row-major order (0,0),(0,1),...,(0,5),(1,1),...,(5,5). Returns an opaque
    pointer (or NULL without GSL). Free with gala_diffusion_grid_free.
*/
#define GALA_DIFF_NFIELDS 27 /* 6 drift + 21 upper-tri tensor */
void *gala_diffusion_grid_alloc(const double *R, int nR, const double *z, int nz,
                                const double *fields);
void gala_diffusion_grid_free(void *grid);

#ifdef __cplusplus
}
#endif

#endif
