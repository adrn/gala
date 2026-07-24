/*
    Implementation of the diffusion RNG, Cholesky, basis rotations, the kick, and the
    builtin diffusion models. GSL is used for the RNG and for 2D interpolation of
    gridded coefficients, all guarded by USE_GSL.
*/
#include "extra_compile_macros.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "diffusion.h"

#if USE_GSL == 1
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_interp2d.h"
#include "gsl/gsl_spline2d.h"
#endif

typedef struct {
#if USE_GSL == 1
    gsl_spline2d *splines[GALA_DIFF_NFIELDS];
    gsl_interp_accel *accR;
    gsl_interp_accel *accz;
#endif
    double R_min, R_max, z_min, z_max;
    int active[GALA_DIFF_NFIELDS];
} grid2d_state;

extern "C" {

// ---- RNG ----

void *gala_diffusion_rng_alloc(unsigned long long seed) {
#if USE_GSL == 1
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, (unsigned long)seed);
    return (void *)r;
#else
    (void)seed;
    return NULL;
#endif
}

void gala_diffusion_rng_set(void *rng, unsigned long long seed) {
#if USE_GSL == 1
    if (rng != NULL) gsl_rng_set((gsl_rng *)rng, (unsigned long)seed);
#else
    (void)rng; (void)seed;
#endif
}

double gala_diffusion_rng_gaussian(void *rng) {
#if USE_GSL == 1
    return gsl_ran_ugaussian((gsl_rng *)rng);
#else
    (void)rng;
    return 0.0;
#endif
}

void gala_diffusion_rng_free(void *rng) {
#if USE_GSL == 1
    if (rng != NULL) gsl_rng_free((gsl_rng *)rng);
#else
    (void)rng;
#endif
}

// ---- Cholesky (lower-triangular, clamps non-PD directions to zero) ----

int gala_diffusion_cholesky(const double *M, int n, double *L) {
    for (int i = 0; i < n * n; i++) L[i] = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = M[i * n + j];
            for (int k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                L[i * n + j] = (s > 0.0) ? sqrt(s) : 0.0;
            } else {
                double Ljj = L[j * n + j];
                L[i * n + j] = (Ljj != 0.0) ? (s / Ljj) : 0.0;
            }
        }
    }
    return 0;
}

// ---- Basis rotation (model basis -> Cartesian) ----

void gala_diffusion_build_rotation(int basis, const double *x, const double *v,
                                   double *Qpos, double *Qvel) {
    (void)v;
    for (int i = 0; i < 9; i++) { Qpos[i] = 0.0; Qvel[i] = 0.0; }
    for (int i = 0; i < 3; i++) { Qpos[i * 3 + i] = 1.0; Qvel[i * 3 + i] = 1.0; }

    if (basis == GALA_DIFF_BASIS_CYLINDRICAL) {
        double phi = atan2(x[1], x[0]);
        double c = cos(phi), s = sin(phi);
        // columns are (R_hat, phi_hat, z_hat): maps cylindrical components to Cartesian
        double Q[9] = {c, -s, 0.0,
                       s,  c, 0.0,
                       0.0, 0.0, 1.0};
        for (int i = 0; i < 9; i++) { Qpos[i] = Q[i]; Qvel[i] = Q[i]; }
    }
    // GALA_DIFF_BASIS_CARTESIAN => identity (already set)
    // future: GALA_DIFF_BASIS_VELOCITY_ALIGNED => build Qvel from normalized v
}

// ---- The kick ----

void diffusion_kick_increments(double t, double dt, double *x, double *v,
                               double *dx, double *dv, void *state, void *rng) {
    CDiffusion *diff = (CDiffusion *)state;
    double mu[GALA_DIFF_PS], M[GALA_DIFF_PS * GALA_DIFF_PS];
    double L[GALA_DIFF_PS * GALA_DIFF_PS], RL[GALA_DIFF_PS * GALA_DIFF_PS];
    double rmu[GALA_DIFF_PS], xi[GALA_DIFF_PS];
    double Qpos[9], Qvel[9];
    int a, b, c;

    // 1. local coefficients (model basis)
    diff->func(t, diff->parameters, x, v, mu, M, diff->state);

    // 2. factor: L L^T = D (or L = B directly)
    if (diff->returns_factor == 0) {
        gala_diffusion_cholesky(M, GALA_DIFF_PS, L);
    } else {
        memcpy(L, M, GALA_DIFF_PS * GALA_DIFF_PS * sizeof(double));
    }

    // 3. rotate to Cartesian: RL = blockdiag(Qpos, Qvel) @ L ; rmu = blockdiag @ mu
    gala_diffusion_build_rotation(diff->basis, x, v, Qpos, Qvel);
    for (a = 0; a < 3; a++) {
        for (b = 0; b < GALA_DIFF_PS; b++) {
            double s0 = 0.0, s1 = 0.0;
            for (c = 0; c < 3; c++) {
                s0 += Qpos[a * 3 + c] * L[c * GALA_DIFF_PS + b];
                s1 += Qvel[a * 3 + c] * L[(3 + c) * GALA_DIFF_PS + b];
            }
            RL[a * GALA_DIFF_PS + b] = s0;
            RL[(3 + a) * GALA_DIFF_PS + b] = s1;
        }
        double m0 = 0.0, m1 = 0.0;
        for (c = 0; c < 3; c++) {
            m0 += Qpos[a * 3 + c] * mu[c];
            m1 += Qvel[a * 3 + c] * mu[3 + c];
        }
        rmu[a] = m0;
        rmu[3 + a] = m1;
    }

    // 4. draw noise & form increment: dw = rmu*dt + RL @ (sqrt(dt) xi)
    double sdt = sqrt(dt);
    for (b = 0; b < GALA_DIFF_PS; b++) xi[b] = gala_diffusion_rng_gaussian(rng);
    for (a = 0; a < GALA_DIFF_PS; a++) {
        double inc = rmu[a] * dt;
        for (b = 0; b < GALA_DIFF_PS; b++) inc += RL[a * GALA_DIFF_PS + b] * sdt * xi[b];
        if (a < 3) dx[a] = inc;
        else dv[a - 3] = inc;
    }
}

// ---- builtin models ----

void constant_diffusion(double t, double *pars, double *x, double *v,
                        double *mu, double *M, void *state) {
    (void)t; (void)x; (void)v; (void)state;
    for (int i = 0; i < GALA_DIFF_PS; i++) mu[i] = pars[i];
    for (int i = 0; i < GALA_DIFF_PS * GALA_DIFF_PS; i++) M[i] = pars[GALA_DIFF_PS + i];
}

void gridded_cyl_diffusion(double t, double *pars, double *x, double *v,
                           double *mu, double *M, void *state) {
    (void)t; (void)pars; (void)v;
    for (int i = 0; i < GALA_DIFF_PS; i++) mu[i] = 0.0;
    for (int i = 0; i < GALA_DIFF_PS * GALA_DIFF_PS; i++) M[i] = 0.0;
#if USE_GSL == 1
    grid2d_state *g = (grid2d_state *)state;
    double R = sqrt(x[0] * x[0] + x[1] * x[1]);
    double az = fabs(x[2]);
    if (R < g->R_min) R = g->R_min;
    if (R > g->R_max) R = g->R_max;
    if (az < g->z_min) az = g->z_min;
    if (az > g->z_max) az = g->z_max;

    for (int f = 0; f < 6; f++) {
        if (g->active[f])
            mu[f] = gsl_spline2d_eval(g->splines[f], R, az, g->accR, g->accz);
    }
    int f = 6;
    for (int i = 0; i < GALA_DIFF_PS; i++) {
        for (int j = i; j < GALA_DIFF_PS; j++) {
            double val = 0.0;
            if (g->active[f])
                val = gsl_spline2d_eval(g->splines[f], R, az, g->accR, g->accz);
            M[i * GALA_DIFF_PS + j] = val;
            M[j * GALA_DIFF_PS + i] = val;
            f++;
        }
    }
#else
    (void)state;
#endif
}

// ---- gridded state management ----

void *gala_diffusion_grid_alloc(const double *R, int nR, const double *z, int nz,
                                const double *fields) {
#if USE_GSL == 1
    grid2d_state *g = (grid2d_state *)malloc(sizeof(grid2d_state));
    if (g == NULL) return NULL;
    g->R_min = R[0];
    g->R_max = R[nR - 1];
    g->z_min = z[0];
    g->z_max = z[nz - 1];
    g->accR = gsl_interp_accel_alloc();
    g->accz = gsl_interp_accel_alloc();

    const gsl_interp2d_type *T = gsl_interp2d_bicubic;
    double *za = (double *)malloc((size_t)nR * nz * sizeof(double));

    for (int fidx = 0; fidx < GALA_DIFF_NFIELDS; fidx++) {
        const double *slice = fields + (size_t)fidx * nR * nz; // slice[i*nz + j]
        int active = 0;
        for (int k = 0; k < nR * nz; k++) {
            if (slice[k] != 0.0) { active = 1; break; }
        }
        g->active[fidx] = active;
        g->splines[fidx] = NULL;
        if (!active) continue;

        gsl_spline2d *sp = gsl_spline2d_alloc(T, nR, nz);
        for (int i = 0; i < nR; i++)
            for (int j = 0; j < nz; j++)
                gsl_spline2d_set(sp, za, i, j, slice[(size_t)i * nz + j]);
        gsl_spline2d_init(sp, R, z, za, nR, nz);
        g->splines[fidx] = sp;
    }

    free(za);
    return (void *)g;
#else
    (void)R; (void)nR; (void)z; (void)nz; (void)fields;
    return NULL;
#endif
}

void gala_diffusion_grid_free(void *grid) {
#if USE_GSL == 1
    if (grid == NULL) return;
    grid2d_state *g = (grid2d_state *)grid;
    for (int f = 0; f < GALA_DIFF_NFIELDS; f++) {
        if (g->splines[f] != NULL) gsl_spline2d_free(g->splines[f]);
    }
    if (g->accR != NULL) gsl_interp_accel_free(g->accR);
    if (g->accz != NULL) gsl_interp_accel_free(g->accz);
    free(g);
#else
    (void)grid;
#endif
}

} // extern "C"
