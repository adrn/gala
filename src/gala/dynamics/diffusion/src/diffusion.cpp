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
#include "gsl/gsl_errno.h"

/* restores the GSL error handler on scope exit (same pattern as staeckel_fudge.cpp) */
struct GSLErrorHandlerOff {
    gsl_error_handler_t *old;
    GSLErrorHandlerOff() : old(gsl_set_error_handler_off()) {}
    ~GSLErrorHandlerOff() { gsl_set_error_handler(old); }
};
#endif

/* Read-only after construction (splines are evaluated with NULL accelerators), so one
   model can be shared by concurrent integrations. */
typedef struct {
#if USE_GSL == 1
    gsl_spline2d *splines[GALA_DIFF_NFIELDS];
#endif
    double R_min, R_max, z_min, z_max;
    int active[GALA_DIFF_NFIELDS];
} grid2d_state;

extern "C" {

// ---- RNG ----

void *gala_diffusion_rng_alloc(unsigned long long seed) {
#if USE_GSL == 1
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, (unsigned long)seed); // NB: mt19937 uses only the low 32 bits
    return (void *)r;
#else
    (void)seed;
    return NULL;
#endif
}

double gala_diffusion_rng_gaussian(void *rng) {
#if USE_GSL == 1
    // ziggurat: much faster than gsl_ran_ugaussian's polar method (no log per draw)
    return gsl_ran_gaussian_ziggurat((gsl_rng *)rng, 1.0);
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

void gala_diffusion_cholesky(const double *M, int n, double *L) {
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
}

// ---- The kick ----

void diffusion_kick_increments(double t, double dt, double *x, double *v,
                               double *dx, double *dv, void *state, void *rng) {
    CDiffusion *diff = (CDiffusion *)state;
    double mu[GALA_DIFF_PS], M[GALA_DIFF_PS * GALA_DIFF_PS];
    double L[GALA_DIFF_PS * GALA_DIFF_PS], xi[GALA_DIFF_PS], inc[GALA_DIFF_PS];
    int a, b;

    // 1. local coefficients (model basis)
    diff->func(t, diff->parameters, x, v, mu, M, diff->state);

    // 2. factor: L L^T = D (or L = B directly)
    if (diff->returns_factor == 0) {
        gala_diffusion_cholesky(M, GALA_DIFF_PS, L);
    } else {
        memcpy(L, M, GALA_DIFF_PS * GALA_DIFF_PS * sizeof(double));
    }

    // 3. Euler-Maruyama increment in the model's coordinates: mu*dt + L sqrt(dt) xi
    double sdt = sqrt(dt);
    for (b = 0; b < GALA_DIFF_PS; b++) xi[b] = gala_diffusion_rng_gaussian(rng);
    for (a = 0; a < GALA_DIFF_PS; a++) {
        inc[a] = mu[a] * dt;
        for (b = 0; b < GALA_DIFF_PS; b++) inc[a] += L[a * GALA_DIFF_PS + b] * sdt * xi[b];
    }

    // 4. apply in those coordinates and return the Cartesian change
    double R = sqrt(x[0] * x[0] + x[1] * x[1]);
    if (diff->basis == GALA_DIFF_BASIS_CYLINDRICAL && R > 0.0) {
        // (R, phi_hat = R dphi, z, v_R, v_phi, v_z): update the curvilinear coordinates
        // (velocities stay attached to the local frame at the new azimuth), as in the
        // Winter-Granic et al. update rules. Rotating increments into a fixed Cartesian
        // frame instead would add v_phi * dphi to v_R.
        double c = x[0] / R, s = x[1] / R;
        double vR = c * v[0] + s * v[1], vp = -s * v[0] + c * v[1];
        double R1 = R + inc[0], dphi = inc[1] / R;
        double c1 = c * cos(dphi) - s * sin(dphi), s1 = s * cos(dphi) + c * sin(dphi);
        double vR1 = vR + inc[3], vp1 = vp + inc[4];
        // exact zeros when the position isn't kicked (leapfrog then skips re-evaluating
        // the gradient; R * c can differ from x[0] by round-off)
        int moved = (inc[0] != 0.0 || inc[1] != 0.0);
        dx[0] = moved ? R1 * c1 - x[0] : 0.0;
        dx[1] = moved ? R1 * s1 - x[1] : 0.0;
        dx[2] = inc[2];
        dv[0] = vR1 * c1 - vp1 * s1 - v[0];
        dv[1] = vR1 * s1 + vp1 * c1 - v[1];
        dv[2] = inc[5];
    } else {
        // Cartesian (and the cylindrical axis, where the basis is undefined)
        for (a = 0; a < 3; a++) {
            dx[a] = inc[a];
            dv[a] = inc[3 + a];
        }
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
    // Outside the grid there is no modeled diffusion: return zero coefficients
    // (mu and M are already zeroed above). This also avoids a GSL out-of-range
    // error from gsl_spline2d_eval.
    // (written as !(in range) so NaN positions also land here)
    if (!(R >= g->R_min && R <= g->R_max && az >= g->z_min && az <= g->z_max)) {
        return;
    }

    for (int f = 0; f < 6; f++) {
        if (g->active[f])
            mu[f] = gsl_spline2d_eval(g->splines[f], R, az, NULL, NULL);
    }
    int f = 6;
    for (int i = 0; i < GALA_DIFF_PS; i++) {
        for (int j = i; j < GALA_DIFF_PS; j++) {
            double val = 0.0;
            if (g->active[f])
                val = gsl_spline2d_eval(g->splines[f], R, az, NULL, NULL);
            M[i * GALA_DIFF_PS + j] = val;
            M[j * GALA_DIFF_PS + i] = val;
            f++;
        }
    }

    // The grid is in |z|: below the midplane, reflect the components that are odd
    // under z -> -z (mu_z, mu_vz, and tensor entries pairing exactly one of z, v_z
    // with an in-plane component). Assumes the model is mirror-symmetric about z = 0.
    if (x[2] < 0.0) {
        mu[2] = -mu[2];
        mu[5] = -mu[5];
        for (int i = 0; i < GALA_DIFF_PS; i++) {
            int oi = (i == 2 || i == 5);
            for (int j = 0; j < GALA_DIFF_PS; j++) {
                if (oi != (j == 2 || j == 5)) M[i * GALA_DIFF_PS + j] = -M[i * GALA_DIFF_PS + j];
            }
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
    GSLErrorHandlerOff guard;
    grid2d_state *g = (grid2d_state *)malloc(sizeof(grid2d_state));
    if (g == NULL) return NULL;
    for (int f = 0; f < GALA_DIFF_NFIELDS; f++) g->splines[f] = NULL;
    g->R_min = R[0];
    g->R_max = R[nR - 1];
    g->z_min = z[0];
    g->z_max = z[nz - 1];

    const gsl_interp2d_type *T = gsl_interp2d_bicubic;
    double *za = (double *)malloc((size_t)nR * nz * sizeof(double));
    if (za == NULL) { free(g); return NULL; }

    for (int fidx = 0; fidx < GALA_DIFF_NFIELDS; fidx++) {
        const double *slice = fields + (size_t)fidx * nR * nz; // slice[i*nz + j]
        int active = 0;
        for (int k = 0; k < nR * nz; k++) {
            if (slice[k] != 0.0) { active = 1; break; }
        }
        g->active[fidx] = active;
        if (!active) continue;

        gsl_spline2d *sp = gsl_spline2d_alloc(T, nR, nz);
        if (sp == NULL) { free(za); gala_diffusion_grid_free(g); return NULL; }
        g->splines[fidx] = sp;
        for (int i = 0; i < nR; i++)
            for (int j = 0; j < nz; j++)
                gsl_spline2d_set(sp, za, i, j, slice[(size_t)i * nz + j]);
        if (gsl_spline2d_init(sp, R, z, za, nR, nz) != GSL_SUCCESS) {
            free(za); gala_diffusion_grid_free(g); return NULL;
        }
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
    free(g);
#else
    (void)grid;
#endif
}

} // extern "C"
