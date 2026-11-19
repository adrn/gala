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
#include "gsl/gsl_eigen.h"

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

// ---- Cholesky (lower-triangular) ----

/* Plain Cholesky that zeroes non-positive pivots. Returns 1 if a pivot is clearly
   negative, i.e. M is indefinite: then the clamped L can be far from M (a tiny pivot
   divides the next column), so the caller must not use it. */
static int cholesky_clamped(const double *M, int n, double *L) {
    int indefinite = 0;
    for (int i = 0; i < n * n; i++) L[i] = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = M[i * n + j];
            for (int k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (s < -1e-10 * fabs(M[i * n + i])) indefinite = 1;
                L[i * n + j] = (s > 0.0) ? sqrt(s) : 0.0;
            } else {
                double Ljj = L[j * n + j];
                L[i * n + j] = (Ljj != 0.0) ? (s / Ljj) : 0.0;
            }
        }
    }
    return indefinite;
}

/* Factor L L^T = M. An indefinite M (e.g. from interpolating a tensor component-wise
   between PSD grid nodes) is first replaced by its nearest PSD matrix (negative
   eigenvalues set to zero), which changes M by at most |lambda_min| in norm. */
void gala_diffusion_cholesky(const double *M, int n, double *L) {
    if (!cholesky_clamped(M, n, L)) return;
#if USE_GSL == 1
    double A[GALA_DIFF_PS * GALA_DIFF_PS], V[GALA_DIFF_PS * GALA_DIFF_PS];
    double lam[GALA_DIFF_PS], P[GALA_DIFF_PS * GALA_DIFF_PS];
    if (n > GALA_DIFF_PS) return;
    memcpy(A, M, (size_t)n * n * sizeof(double));
    gsl_matrix_view Am = gsl_matrix_view_array(A, n, n);
    gsl_matrix_view Vm = gsl_matrix_view_array(V, n, n);
    gsl_vector_view lv = gsl_vector_view_array(lam, n);
    gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(n);
    if (w == NULL) return;
    gsl_eigen_symmv(&Am.matrix, &lv.vector, &Vm.matrix, w);
    gsl_eigen_symmv_free(w);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                if (lam[k] > 0.0) sum += V[i * n + k] * lam[k] * V[j * n + k];
            P[i * n + j] = sum;
        }
    }
    cholesky_clamped(P, n, L);
#endif
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

// ---- action space -> (x, v) mapping (Winter-Granic et al. App. B) ----

void gala_diffusion_epicyclic_map(const double *epi, const double *A_J, const double *D_J,
                                  const double *J, const double *cyl, double *mu,
                                  double *M) {
    const double Rg = epi[0], Om = epi[1], ka = epi[2], qR = epi[4];
    const double JR = J[0], Jp = J[1], Jz = J[2];
    const double DRR = D_J[0], DPP = D_J[1], DZZ = D_J[2];
    const double R = cyl[0], vR = cyl[1], vp = cyl[2], z = cyl[3], vz = cyl[4];

    const double g = 2.0 * Om / ka, g2 = g * g, g3 = g2 * g, g4 = g2 * g2;
    const double q = 2.0 - 2.0 / g2;
    const double vc = Rg * Om;
    const double h = 1.0 - g2 / 2.0;

    const double C = A_J[0] / (2.0 * JR) - DRR / (4.0 * JR * JR);
    const double Cz = A_J[2] / (2.0 * Jz) - DZZ / (4.0 * Jz * Jz);

    double D_RR = (g * Jp / (4.0 * JR) * DRR + g4 / 4.0 * DPP) / (vc * vc);
    double D_Rvp = -g / (4.0 * JR) * DRR + g2 / (2.0 * Jp) * h * DPP;
    double D_vpvp = vc * vc * (g / (4.0 * JR * Jp) * DRR + h * h * DPP / (Jp * Jp));
    double D_vRvR = vc * vc / g * DRR / (JR * Jp);
    double D_pvR = g / 2.0 * vc * DRR / (JR * Jp); // phi in radians
    double D_pp = g3 / 4.0 * DRR / (JR * Jp);
    double D_zz = DZZ / (2.0 * epi[3] * Jz);
    double D_vzvz = epi[3] * DZZ / (2.0 * Jz);

    double A_R = g2 * Rg / (2.0 * Jp) * A_J[1] + (R - Rg) * C
                 + g4 * Rg / (8.0 * Jp * Jp) * (2.0 * qR - q - 1.0) * DPP;

    for (int i = 0; i < GALA_DIFF_PS * GALA_DIFF_PS; i++) M[i] = 0.0;
    // cylindrical basis ordering (R, phi, z, v_R, v_phi, v_z); phi -> physical arc R*phi
    mu[0] = A_R;
    mu[1] = R * g * vR / (Rg * ka) * C;
    mu[2] = z * Cz;
    mu[3] = vR * C;
    mu[4] = (A_J[1] - vp * A_R - D_Rvp) / R;
    mu[5] = vz * Cz;
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
   tabulated rotation curve (or retrograde), in which case nothing is filled.
   epi = (R_g, Omega, kappa, nu, q_R); cyl = (R, v_R, v_phi, z, v_z). */
static int epicyclic_actions(const epicyclic_state *s, const double *x, const double *v,
                             double J_floor, double *epi, double *J, double *cyl) {
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
    double ka = exp(gsl_spline_eval(s->lnkappa, lnRg, NULL));
    double nu = exp(gsl_spline_eval(s->lnnu, lnRg, NULL));
    epi[0] = Rg;
    epi[1] = exp(gsl_spline_eval(s->lnOmega, lnRg, NULL));
    epi[2] = ka;
    epi[3] = nu;
    // q_R = -dln kappa/dln R (1 for a flat rotation curve)
    epi[4] = -gsl_spline_eval_deriv(s->lnkappa, lnRg, NULL);

    double dR = R - Rg;
    J[0] = fmax((vR * vR + ka * ka * dR * dR) / (2.0 * ka), J_floor);
    J[1] = Lz;
    J[2] = fmax((v[2] * v[2] + nu * nu * x[2] * x[2]) / (2.0 * nu), J_floor);
    cyl[0] = R;
    cyl[1] = vR;
    cyl[2] = Lz / R;
    cyl[3] = x[2];
    cyl[4] = v[2];
    return 1;
#else
    (void)s; (void)x; (void)v; (void)J_floor; (void)epi; (void)J; (void)cyl;
    return 0;
#endif
}

/*
    ISM-driven diffusion (Modak et al. 2026 fits as used by Winter-Granic et al. 2027,
    following their reference script). The fits are rescaled to each orbit's guiding
    radius (R_g, kappa, nu), suppressed for vertically hot orbits by the gas-disc
    thickness, and multiplied by factors for the ISM surface density, fluctuation scale
    and correlation time (computed in Python). Regimes switch at fixed breaks (in kpc
    km/s). Drifts A_i = 1/2 dD_ii/dJ_i; A_Jphi = 0.
    pars = [amp2, J_unit, D_unit, J_floor, longwave, kappa_ref, nu_ref, R_ref, H_gas,
            k_star, f_low, f_R_high, f_P_high]
    (all in the integrator's units; D_unit = (kpc km/s)^2/Gyr, J_unit = kpc km/s).
*/
void ism_diffusion_winter27(double t, double *pars, double *x, double *v,
                            double *mu, double *M, void *state) {
    (void)t;
    for (int i = 0; i < GALA_DIFF_PS; i++) mu[i] = 0.0;
    for (int i = 0; i < GALA_DIFF_PS * GALA_DIFF_PS; i++) M[i] = 0.0;

    const double amp2 = pars[0], Ju = pars[1], Du = pars[2], J_floor = pars[3];
    const int longwave = pars[4] != 0.0;
    const double kappa_ref = pars[5], nu_ref = pars[6], R_ref = pars[7];
    const double H_gas = pars[8], k_star = pars[9];
    const double f_low = pars[10], f_R_high = pars[11], f_P_high = pars[12];

    double epi[5], J[3], cyl[5];
    if (!epicyclic_actions((const epicyclic_state *)state, x, v, J_floor, epi, J, cyl))
        return;

    // local environment at the guiding radius
    const double nu = epi[3];
    const double kr = epi[2] / kappa_ref, rr = epi[0] / R_ref, nr = nu / nu_ref;

    const double jR = J[0] / Ju, jz = J[2] / Ju; // dimensionless actions
    const double inf = INFINITY;
    const double JR_break = longwave ? inf : 2.4, Jp_break = longwave ? inf : 0.72;
    const double JRz_break = longwave ? inf : 1.5, Jz_break = 2.3;

    double D_J[3], A_J[3] = {0.0, 0.0, 0.0}; // in D_unit and D_unit / J_unit

    // D_JRJR: 9.3 jR / kr (long wavelength) | 35 jR^-1/2 kr^1/2 (short)
    const int hiR = jR > JR_break, hiP = jR > Jp_break;
    if (hiR) {
        D_J[0] = 35.0 / sqrt(jR) * sqrt(kr) * f_R_high;
        A_J[0] = -0.25 * D_J[0] / jR;
    } else {
        D_J[0] = 9.3 * jR / kr * f_low;
        A_J[0] = 0.5 * 9.3 / kr * f_low;
    }
    // D_JpJp: 6900 rr^2 | 5900 jR^-1/2 rr^2 kr^1/2 (J_phi dependence not used => A_Jp = 0)
    D_J[1] = hiP ? 5900.0 / sqrt(jR) * rr * rr * sqrt(kr) * f_P_high
                 : 6900.0 * rr * rr * f_low;

    // the thickness of the gas disc suppresses planar forcing of vertically hot orbits
    const double KH = k_star * H_gas;
    const double Jv = fmax(2.0 * J[2] / (nu * H_gas * H_gas), 0.0);
    const double pre = (1.0 + 0.71 * KH) * (1.0 + 0.71 * KH);
    const double xi = pre / (1.0 + (pre - 1.0) * (1.0 + cbrt(KH) * pow(Jv, 0.75)));
    const double sR = xi / (hiR ? 0.75 : 0.94);
    D_J[0] *= sR;
    A_J[0] *= sR;
    D_J[1] *= xi / (hiP ? 0.84 : 0.98);

    // D_JzJz: 0.29 jz / nr | 0.35 jz jR^-1/2 kr^1/2 / nr | 1.2 jz^-1/2 jR^-1/2 nr^1/2 kr^1/2
    if (!(jR > JRz_break)) {
        D_J[2] = 0.29 * jz / nr;
        A_J[2] = 0.5 * 0.29 / nr;
    } else if (jz > Jz_break) {
        D_J[2] = 1.2 / sqrt(jz * jR) * sqrt(nr * kr);
        A_J[2] = -0.25 * D_J[2] / jz;
    } else {
        D_J[2] = 0.35 * jz / sqrt(jR) * sqrt(kr) / nr;
        A_J[2] = 0.5 * D_J[2] / jz;
    }

    const double k = amp2 * Du;
    for (int i = 0; i < 3; i++) {
        D_J[i] *= k;
        A_J[i] *= k / Ju;
    }
    gala_diffusion_epicyclic_map(epi, A_J, D_J, J, cyl, mu, M);
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
