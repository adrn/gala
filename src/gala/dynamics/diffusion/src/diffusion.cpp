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
#include "gsl/gsl_spline.h"
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

/* Rotation-curve tables for the epicyclic action-space models, as splines in ln R.
   Read-only after construction (evaluated with NULL accelerators). */
typedef struct {
#if USE_GSL == 1
    gsl_spline *lnOmega; // ln Omega(ln R)
    gsl_spline *lnkappa; // ln kappa(ln R)
    gsl_spline *lnnu;    // ln nu(ln R)
    gsl_spline *lnRg;    // ln R_g(ln L_z), the inverse of L_z = R v_c(R)
#endif
    double lnR_min, lnR_max, lnLz_min, lnLz_max;
} epicyclic_state;

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
        double R = sqrt(x[0] * x[0] + x[1] * x[1]);
        if (R == 0.0) return; // R_hat undefined on the axis: leave identity
        double c = x[0] / R, s = x[1] / R;
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
#else
    (void)state;
#endif
}

// ---- action space -> (x, v) mapping (Winter-Granic et al. App. B, general q_R) ----

void gala_diffusion_epicyclic_map(const double *epi, const double *J, const double *A_J,
                                  const double *D_J, double R, double vz,
                                  double *mu, double *M) {
    const double Rg = epi[0], Om = epi[1], ka = epi[2], nu = epi[3];
    const double qR = epi[4], qz = epi[5], dqR = epi[6];
    const double JR = J[0], Jp = J[1], Jz = J[2];
    const double AR = A_J[0], Ap = A_J[1], Az = A_J[2];
    const double DRR = D_J[0], DPP = D_J[1], DZZ = D_J[2], DPR = D_J[3], DZP = D_J[4];

    const double g = 2.0 * Om / ka, g2 = g * g, g3 = g2 * g, g4 = g2 * g2;
    const double eps = (2.0 / g) * JR / Jp;
    const double V = Rg * Om;
    const double rt2 = sqrt(2.0);

    for (int i = 0; i < GALA_DIFF_PS; i++) mu[i] = 0.0;
    for (int i = 0; i < GALA_DIFF_PS * GALA_DIFF_PS; i++) M[i] = 0.0;

    // drifts
    const double B = (1.0 - g2 / 2.0) * (1.0 - 2.0 * eps) + eps * (4.0 + g2 * (qR - 2.0));
    const double c1 = 1.0 + g2 / 2.0 * (qR - 1.0);
    const double c2 = qR - 2.0 + 2.0 / g2;
    double A_R = g2 * Rg / (2.0 * Jp) * Ap
                 + Rg / (2.0 * Jp * Jp) * (g2 / 2.0 + g4 / 4.0 * (2.0 * qR - 3.0)) * DPP;
    double A_vp = V * (-2.0 * eps * AR / JR + B * Ap / Jp + 2.0 * eps * c1 * DPR / (Jp * JR)
                       + 0.5 * (-g2 / 2.0 * B - g4 / 2.0 * c2 * (1.0 - 2.0 * eps)
                                + 2.0 * eps * (2.0 + g2 / 2.0 * (qR - 2.0)) * c1
                                + eps * g4 * c2 * (qR - 2.0) + eps * g4 / 2.0 * dqR)
                             * DPP / (Jp * Jp));
    double A_vz = vz * (Az / Jz - qz * g2 / 2.0 * Ap / Jp - DZZ / (Jz * Jz)
                        + qz * DZP / (Jz * Jp) + qz * (1.0 - qz) / 2.0 * DPP / (Jp * Jp));

    // diffusion (phi entries in radians here)
    const double c3 = 1.0 + g2 / 4.0 * (qR - 4.0);
    const double c4 = 1.0 + 3.0 * g2 / 4.0 * (qR - 2.0);
    double D_RR = (DRR / eps + g3 * qR / 4.0 * DPR
                   + g4 / 4.0 * (1.0 + qR * qR * g2 * eps / 16.0) * DPP) / (V * V);
    double D_Rvp = -DRR / (2.0 * rt2 * JR) + (-15.0 / 4.0 + 9.0 / 8.0 * eps) * DPR / (rt2 * Jp)
                   + eps / (8.0 * Jp) * DPP;
    double D_vpvp = V * V * (g2 * eps / 16.0 * DRR / (JR * JR)
                             + ((1.0 - g2 / 2.0) * (1.0 - g2 / 2.0) + g2 * eps / 4.0 * c3 * c3)
                                   * DPP / (Jp * Jp)
                             + g2 * eps / 4.0 * c3 * DPR / (Jp * JR));
    double D_vRvR = V * V * eps / 4.0 * (g4 * qR * qR / 4.0 * DPP / (Jp * Jp)
                                         - 2.0 * g2 * qR * DPR / (JR * Jp)
                                         + 4.0 * DRR / (JR * JR));
    double D_pvR = eps * V * g2 / 8.0 * (qR * g2 / 2.0 * DPP / (Jp * Jp)
                                         - (1.0 + qR * g2 / 2.0) * DPR / (JR * Jp)
                                         + qR * g2 / 2.0 * DRR / (JR * JR));
    double D_pp = g4 * eps / 16.0 * (4.0 * c4 * DPP / (Jp * Jp)
                                     - 4.0 * c4 * c4 * DPR / (JR * Jp)
                                     + DRR / (JR * JR));
    double D_zz = (DZZ + qz * qz / 4.0 * Jz * Jz / (Jp * Jp) * DPP + qz * Jz / Jp * DZP)
                  / (nu * Jz);
    double D_vzvz = nu / Jz * (DZZ + qz * qz / 4.0 * Jz * Jz / (Jp * Jp) * DPP
                               - qz * Jz / Jp * DZP);

    // cylindrical basis ordering (R, phi, z, v_R, v_phi, v_z); phi -> physical arc R*phi
    mu[0] = A_R;
    mu[4] = A_vp;
    mu[5] = A_vz;
#define SETM(i, j, val) { M[(i) * GALA_DIFF_PS + (j)] = (val); M[(j) * GALA_DIFF_PS + (i)] = (val); }
    SETM(0, 0, D_RR);
    SETM(0, 4, D_Rvp);
    SETM(4, 4, D_vpvp);
    SETM(3, 3, D_vRvR);
    SETM(1, 3, R * D_pvR);
    SETM(1, 1, R * R * D_pp);
    SETM(2, 2, D_zz);
    SETM(5, 5, D_vzvz);
#undef SETM
}

/* Epicyclic quantities and actions at (x, v). Returns 0 if the orbit is outside the
   tabulated rotation curve (or retrograde), in which case nothing is filled. */
static int epicyclic_actions(const epicyclic_state *s, const double *x, const double *v,
                             double J_floor, double *epi, double *J, double *R_out,
                             double *vz_out) {
#if USE_GSL == 1
    double R = sqrt(x[0] * x[0] + x[1] * x[1]);
    if (!(R > 0.0)) return 0;
    double vR = (x[0] * v[0] + x[1] * v[1]) / R;
    double Lz = x[0] * v[1] - x[1] * v[0];
    if (!(Lz > 0.0)) return 0;
    double lnLz = log(Lz);
    if (!(lnLz >= s->lnLz_min && lnLz <= s->lnLz_max)) return 0;

    double lnRg = gsl_spline_eval(s->lnRg, lnLz, NULL);
    double Rg = exp(lnRg);
    double Om = exp(gsl_spline_eval(s->lnOmega, lnRg, NULL));
    double ka = exp(gsl_spline_eval(s->lnkappa, lnRg, NULL));
    double nu = exp(gsl_spline_eval(s->lnnu, lnRg, NULL));
    epi[0] = Rg;
    epi[1] = Om;
    epi[2] = ka;
    epi[3] = nu;
    // q_R = -dln kappa/dln R and q_z = -dln nu/dln R (both 1 for a flat rotation curve,
    // as the App. B formulas assume; the paper's text definition omits the sign)
    epi[4] = -gsl_spline_eval_deriv(s->lnkappa, lnRg, NULL);
    epi[5] = -gsl_spline_eval_deriv(s->lnnu, lnRg, NULL);
    epi[6] = -gsl_spline_eval_deriv2(s->lnkappa, lnRg, NULL); // dq_R / dln R

    double dR = R - Rg;
    J[0] = fmax((vR * vR + ka * ka * dR * dR) / (2.0 * ka), J_floor);
    J[1] = Lz;
    J[2] = fmax((v[2] * v[2] + nu * nu * x[2] * x[2]) / (2.0 * nu), J_floor);
    *R_out = R;
    *vz_out = v[2];
    return 1;
#else
    (void)s; (void)x; (void)v; (void)J_floor; (void)epi; (void)J; (void)R_out; (void)vz_out;
    return 0;
#endif
}

/*
    ISM-driven diffusion (Modak et al. 2026 solar-neighborhood fits, as quoted in
    Winter-Granic et al. 2027). The asymptotic regimes are joined where they cross
    (min over branches), so D is continuous. Drifts A_i = 1/2 dD_ii/dJ_i (D_J diagonal).
    pars = [scale, J_unit, D_unit, J_floor]; D_unit = (kpc km/s)^2/Gyr, J_unit = kpc km/s
    in the integrator's units; scale multiplies every D and A.
*/
void ism_diffusion_winter27(double t, double *pars, double *x, double *v,
                            double *mu, double *M, void *state) {
    (void)t;
    for (int i = 0; i < GALA_DIFF_PS; i++) mu[i] = 0.0;
    for (int i = 0; i < GALA_DIFF_PS * GALA_DIFF_PS; i++) M[i] = 0.0;

    const double scale = pars[0], Ju = pars[1], Du = pars[2], J_floor = pars[3];
    double epi[7], J[3], R, vz;
    if (!epicyclic_actions((const epicyclic_state *)state, x, v, J_floor, epi, J, &R, &vz))
        return;

    const double jR = J[0] / Ju, jz = J[2] / Ju; // dimensionless actions
    const double k = scale * Du;
    double D_J[5] = {0.0, 0.0, 0.0, 0.0, 0.0}; // JRJR, JpJp, JzJz, JpJR, JzJp
    double A_J[3] = {0.0, 0.0, 0.0};          // in D_unit / J_unit

    // D_JRJR: 9.3 jR (long wavelength) | 6.9 jR^-1/2 (short)
    double a = 9.3 * jR, b = 6.9 / sqrt(jR);
    D_J[0] = fmin(a, b);
    A_J[0] = (a <= b) ? 0.5 * 9.3 : -0.25 * 6.9 * pow(jR, -1.5);

    // D_JpJp: 6900 | 4600 jR^-1/2 (no J_phi dependence => A_Jp = 0)
    D_J[1] = fmin(6900.0, 4600.0 / sqrt(jR));

    // D_JzJz: 0.29 jz | 0.04 jz jR^-1/2 | 0.14 jz^-1/2 jR^-1/2
    double z1 = 0.29 * jz, z2 = 0.04 * jz / sqrt(jR), z3 = 0.14 / sqrt(jz * jR);
    if (z1 <= z2 && z1 <= z3) {
        D_J[2] = z1;
        A_J[2] = 0.5 * 0.29;
    } else if (z2 <= z3) {
        D_J[2] = z2;
        A_J[2] = 0.5 * 0.04 / sqrt(jR);
    } else {
        D_J[2] = z3;
        A_J[2] = -0.25 * z3 / jz;
    }

    for (int i = 0; i < 5; i++) D_J[i] *= k;
    for (int i = 0; i < 3; i++) A_J[i] *= k / Ju;

    gala_diffusion_epicyclic_map(epi, J, A_J, D_J, R, vz, mu, M);
}

void *gala_diffusion_epicyclic_alloc(const double *lnR, const double *lnOmega,
                                     const double *lnkappa, const double *lnnu,
                                     const double *lnLz, int n) {
#if USE_GSL == 1
    GSLErrorHandlerOff guard;
    epicyclic_state *s = (epicyclic_state *)calloc(1, sizeof(epicyclic_state));
    if (s == NULL) return NULL;
    s->lnR_min = lnR[0];
    s->lnR_max = lnR[n - 1];
    s->lnLz_min = lnLz[0];
    s->lnLz_max = lnLz[n - 1];

    gsl_spline **sp[4] = {&s->lnOmega, &s->lnkappa, &s->lnnu, &s->lnRg};
    const double *xs[4] = {lnR, lnR, lnR, lnLz};
    const double *ys[4] = {lnOmega, lnkappa, lnnu, lnR};
    for (int i = 0; i < 4; i++) {
        *sp[i] = gsl_spline_alloc(gsl_interp_cspline, n);
        if (*sp[i] == NULL || gsl_spline_init(*sp[i], xs[i], ys[i], n) != GSL_SUCCESS) {
            gala_diffusion_epicyclic_free(s);
            return NULL;
        }
    }
    return (void *)s;
#else
    (void)lnR; (void)lnOmega; (void)lnkappa; (void)lnnu; (void)lnLz; (void)n;
    return NULL;
#endif
}

void gala_diffusion_epicyclic_free(void *state) {
#if USE_GSL == 1
    if (state == NULL) return;
    epicyclic_state *s = (epicyclic_state *)state;
    gsl_spline *sp[4] = {s->lnOmega, s->lnkappa, s->lnnu, s->lnRg};
    for (int i = 0; i < 4; i++)
        if (sp[i] != NULL) gsl_spline_free(sp[i]);
    free(s);
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
