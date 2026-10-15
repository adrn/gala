/* Staeckel fudge */
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "extra_compile_macros.h"
#include "staeckel_fudge.h"

#if USE_GSL == 1
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>

namespace {

/* ------------------------------------------------------------
    Constants and defaults:
*/

/*
    Gauss-Legendre order ranges (since we are using a fixed GL). The defaults in Agama
    are 10 and 40, but I *think* the change of variables here might allow fewer nodes
    (it's closer to that suggested by Sanders (2012) in the Appendix, which recommends
    10 GL points). I should probably chat with Eugene about this, because I suspect his
    heuristics come from some experiments. The min order corresponds to near-circular
    orbits, max for near-radial.

    TODO: make these configurable?
*/
const int MIN_GL_ORDER = 10;
const int MAX_GL_ORDER = 32;

/* relative tolerance for the integration limits in (lambda, nu) */
// TODO: make this configurable
const double ACCURACY_RANGE = 1e-12;

/* minimum range of nu, lambda that is considered to be non-zero */
// TODO: make this configurable
const double MINIMUM_RANGE = 1e-10;

/* ------------------------------------------------------------
    Helper functions:
*/
inline double sq(double x) { return x * x; }

double wrap_angle(double x) {
    x = fmod(x, 2 * M_PI);
    return x < 0 ? x + 2 * M_PI : x;
}

/*
    Cylindrical (R, z, vR, vz) -> prolate spheroidal (lambda, nu) and time derivatives,
    for focal distance squared D2. lambda and nu are the roots t of R^2/(t - D2) + z^2/t
    = 1, with lambda in [D2, inf) and nu in [-D2, D2] (with the sign of z).
*/
void cyl_to_prolsph(double R, double z, double vR, double vz, double D2,
                    double *lambda, double *nu, double *lambdadot, double *nudot) {
    const double R2 = R * R, z2 = z * z;
    const double signz = z >= 0 ? 1 : -1;
    const double sum = R2 + z2 + D2, dif = R2 + z2 - D2;
    double sqD = sqrt(sq(dif) + 4 * R2 * D2);
    if (z2 == 0)
        sqD = sum;
    if (R2 == 0)
        sqD = fabs(dif);
    double lmd, dmn;  // lambda - D2 and D2 - |nu|
    if (dif >= 0) {
        lmd = 0.5 * (sqD + dif);
        dmn = R2 > 0 ? D2 * R2 / lmd : 0;
    } else {
        dmn = 0.5 * (sqD - dif);
        lmd = D2 * R2 / dmn;
    }
    double absnu = 2 * D2 / (sum + sqD) * z2;
    if (absnu * 2 > D2)
        absnu = D2 - dmn;
    else
        dmn = D2 - absnu;
    *lambda = D2 + lmd;
    *nu = absnu * signz;
    *lambdadot = (R * 2 * *lambda * vR + z * 2 * lmd * vz) / sqD;
    *nudot = (-R * 2 * absnu * vR + z * 2 * dmn * vz) / sqD * signz;
}

/*
    Pick the GL order given an orbit's ~eccentricity (ratio of pericenter/apocenter).
    This picks a higher order for more eccentric orbits. The heuristic comes from Agama.
*/
int integr_order(double R1overR2) {
    if (!(R1overR2 > 0))
        return MAX_GL_ORDER;
    int log2;
    frexp(R1overR2, &log2);
    return std::max(1, std::min(MAX_GL_ORDER, MIN_GL_ORDER - log2));
}

/* ------------------------------------------------------------
    GSL setup and helpers:
*/

/*
    WARNING: CLAUDE wrote this; I audited.
    Set up a fixed-order GL integration table.
*/
gsl_integration_glfixed_table *gl_table(int order) {
    static gsl_integration_glfixed_table *tables[MAX_GL_ORDER + 1] = {NULL};
    if (tables[order] == NULL)
        tables[order] = gsl_integration_glfixed_table_alloc(order);
    return tables[order];
}

/*
    WARNING: CLAUDE wrote this; I audited.
    Root of F on [a, b], or NAN if the endpoints do not bracket a root. Uses GSL's brent inside.
*/
double find_root(gsl_function *F, double a, double b, double reltol) {
    const double fa = GSL_FN_EVAL(F, a), fb = GSL_FN_EVAL(F, b);
    if (!std::isfinite(fa + fb) || (fa < 0 && fb < 0) || (fa > 0 && fb > 0))
        return NAN;
    if (fa == 0)
        return fb == 0 ? 0.5 * (a + b) : a;
    if (fb == 0)
        return b;

    gsl_root_fsolver *solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
    gsl_root_fsolver_set(solver, F, a, b);
    const double tol = reltol * fabs(b - a);
    double root = NAN;
    for (int i = 0; i < 100; i++) {
        if (gsl_root_fsolver_iterate(solver) != GSL_SUCCESS) {
            root = NAN;
            break;
        }
        root = gsl_root_fsolver_root(solver);
        if (gsl_root_test_interval(gsl_root_fsolver_x_lower(solver),
                                   gsl_root_fsolver_x_upper(solver), tol, 0) == GSL_SUCCESS)
            break;
    }
    gsl_root_fsolver_free(solver);
    return root;
}

/* ------------------------------------------------------------
    Main Fudge functionality:
*/

/*
    Container for tracking a given point in prolate spheroidal coords and integrals of motion.
*/
struct Fudge {
    CPotential *pot;
    double t;
    double D2;  // Delta^2
    double lambda, nu, lambdadot, nudot, phi;
    double E, Lz, I3, flambda, fnu;
};

/*
    Potential and cylindrical derivatives (dR, dz) and (dR2, dz2, dRdz).
*/
double eval_potential(const Fudge &f, double R, double z, double *grad, double *hess) {
    double q[3] = {R, 0., z};  // turn R,z -> (x,y,z)
    double Phi = c_potential(f.pot, f.t, q);
    if (grad) {
        double g[3];
        c_gradient(f.pot, 1, f.t, q, g);
        grad[0] = g[0];
        grad[1] = g[2];
    }
    if (hess) {
        double h[9];
        c_hessian(f.pot, f.t, q, h);
        hess[0] = h[0];
        hess[1] = h[8];
        hess[2] = h[2];
    }
    return Phi;
}

/*
    This is the main f(tau) auxiliary function (and derivatives) that is used to compute
    integrals of motion and other internals for the Fudge machinery. See Sanders (2012)
    (Section 2) for much more detail.
*/
void F_tau(const Fudge &f, double tau, double *val, double *der = NULL, double *der2 = NULL) {
    double lambda, nu, I, mult;

    const bool is_lambda = tau >= f.D2;
    if (is_lambda) {
        lambda = tau;
        nu = fabs(f.nu);
        mult = lambda - nu;
        I = f.I3 - f.fnu;
    } else {
        lambda = f.lambda;
        nu = tau;
        mult = nu - lambda;
        I = f.I3 - f.flambda;
    }

    // prolate spheroidal -> cylindrical (with nu >= 0, so z >= 0)
    const double lmd = lambda - f.D2;
    const double nmd = nu - f.D2;
    const double R = sqrt(fmax(0., lmd * (1 - nu / f.D2)));
    const double z = sqrt(lambda * nu / f.D2);

    double grad[2], hess[3];
    const double Phi = eval_potential(f, R, z, (der || der2) ? grad : NULL, der2 ? hess : NULL);
    const double tmd = tau - f.D2;
    if (val)
        *val = (tmd != 0 ? (f.E * tau - I - Phi * mult) * tmd : 0) - sq(f.Lz) / 2 * tau;
    if (!(der || der2))
        return;

    // Jacobian and Hessian terms of the cylindrical/prolate coord transform
    double dRdtau, dzdtau, d2Rdtau2, d2zdtau2;
    if (is_lambda) {
        dRdtau = 0.5 * R / lmd;
        dzdtau = 0.5 * z / lambda;
        d2Rdtau2 = -0.25 * R / sq(lmd);
        d2zdtau2 = -0.25 * z / sq(lambda);
    } else {
        dRdtau = 0.5 * R / nmd;
        dzdtau = 0.5 * z / nu;
        d2Rdtau2 = -0.25 * R / sq(nmd);
        d2zdtau2 = -0.25 * z / sq(nu);
    }
    const double dPhidtau = grad[0] * dRdtau + grad[1] * dzdtau;
    if (der)
        *der = f.E * (tau + tmd) - sq(f.Lz) / 2 - I - (mult + tmd) * Phi -
               (tmd != 0 ? tmd * mult * dPhidtau : 0);
    if (der2) {
        const double d2Phidtau2 = hess[0] * sq(dRdtau) + hess[1] * sq(dzdtau) +
                                  2 * hess[2] * dRdtau * dzdtau + grad[0] * d2Rdtau2 +
                                  grad[1] * d2zdtau2;
        *der2 = 2 * (f.E - Phi - (mult + tmd) * dPhidtau) - tmd * mult * d2Phidtau2;
    }
}

/*
    This is a wrapper with a single void pointer so we can pass to GSL integration.
*/
double F_gsl(double tau, void *params) {
    double val;
    F_tau(*(const Fudge *)params, tau, &val);
    return val;
}

/*
    This container and function handle ...

    F(tau) / tau^2 with tau = tau0 exp(s / (1 - s)), to bracket lambda_max on [tau0,
    inf) with s in [0, 1]; tends to E as tau -> inf
*/
struct SemiInf {
    const Fudge *f;
    double tau0;
};

double F_semiinf_gsl(double s, void *params) {
    const SemiInf *p = (const SemiInf *)params;
    const double tau = p->tau0 * exp(s / (1 - s));
    if (!(tau < 1e100))
        return p->f->E;
    double val;
    F_tau(*p->f, tau, &val);
    return val / sq(tau);
}

/*
    WARNING: CLAUDE helped write this to make my sloppy initial version more consistent
    with Agama's implementation.
    This is a major internal work function for the Staeckel Fudge that determines the
    integral limits for the prolate spheroidal coordinates.
*/
struct Limits {
    double lambda_min, lambda_max, nu_max;
    int order;
};

Limits find_limits(const Fudge &f) {
    Limits lim;
    const double D2 = f.D2;
    gsl_function F = {&F_gsl, (void *)&f};

    double f_zero, f_lambda, df_lambda, d2f_lambda;
    F_tau(f, 0., &f_zero);
    F_tau(f, f.lambda, &f_lambda, &df_lambda, &d2f_lambda);
    lim.nu_max = lim.lambda_min = lim.lambda_max = NAN;
    double nu_upper = D2;      // upper bound for locating nu_max
    double lambda_lower = D2;  // lower bound for locating lambda_min

    if (f.Lz == 0) {
        // F(Delta^2) = -Lz^2/2 = 0: may be either a tube or a box orbit in the meridional plane
        const double deltaminus = D2 * (1 - 1e-15), deltaplus = D2 * (1 + 1e-15);
        double fm, fp;
        F_tau(f, deltaminus, &fm);
        F_tau(f, deltaplus, &fp);
        if (fm < 0) {  // box orbit
            if (f_zero > 0)
                nu_upper = deltaminus;
            else
                lim.nu_max = 0;
        } else
            lim.nu_max = D2;
        if (fp < 0)  // tube orbit
            lambda_lower = deltaplus;
        else
            lim.lambda_min = D2;
    }

    if (!std::isfinite(lim.nu_max)) {
        if (f_zero > 0)
            lim.nu_max = find_root(&F, fabs(f.nu), nu_upper, ACCURACY_RANGE);
        if (!std::isfinite(lim.nu_max))
            // f(nu) was very slightly negative, or f(0) <= 0: nu is the upper boundary
            lim.nu_max = fabs(f.nu);
    }

    // lambda_pos is a point well inside the interval where F > 0, so that
    // [Delta^2, lambda_pos] and [lambda_pos, inf) firmly bracket lambda_min and lambda_max.
    // Due to roundoff, F(lambda) may be slightly negative or tiny, so we may need to nudge it.
    double lambda_pos = f.lambda;
    const double dx_to_root = -f_lambda / df_lambda;
    if (f_lambda <= 0 || fabs(dx_to_root) < f.lambda * MINIMUM_RANGE) {
        // at the edge of the allowed interval: assign one endpoint immediately
        if (df_lambda >= 0)
            lim.lambda_min = f.lambda;
        if (df_lambda <= 0) {
            lim.lambda_max = f.lambda;
            if (f.lambda == D2)
                lim.lambda_min = D2;
        }

        // at or near a shell orbit, F is a downward parabola whose roots nearly coincide
        bool near_shell = d2f_lambda < 0 &&
                          sq(df_lambda) - 2 * f_lambda * d2f_lambda -
                                  sq(f.lambda * MINIMUM_RANGE * d2f_lambda) < 0;

        // F'' is undefined at lambda = Delta^2, so shift slightly toward increasing F and retest
        const double safe_offset =
            fmin(0.5 * fabs(df_lambda / d2f_lambda), f.lambda * MINIMUM_RANGE);
        lambda_pos += fmax(dx_to_root, safe_offset) * (df_lambda >= 0 ? 1 : -1);
        double f_lampos, df_lampos, d2f_lampos;
        F_tau(f, lambda_pos, &f_lampos, &df_lampos, &d2f_lampos);
        near_shell |= d2f_lampos < 0 &&
                      sq(df_lampos) - 2 * f_lampos * d2f_lampos -
                              sq(f.lambda * MINIMUM_RANGE * d2f_lampos) < 0;

        // cancellation in F(lambda) can hide a near-shell orbit, so also accept F < 0 here
        if (near_shell || f_lampos < 0)
            lim.lambda_min = lim.lambda_max = f.lambda;
    }
    if (!std::isfinite(lim.lambda_min))
        lim.lambda_min = find_root(&F, lambda_lower, lambda_pos, ACCURACY_RANGE);
    if (!std::isfinite(lim.lambda_max)) {
        SemiInf params = {&f, lambda_pos};
        gsl_function Fs = {&F_semiinf_gsl, &params};
        const double s = find_root(&Fs, 0., 1., ACCURACY_RANGE);
        lim.lambda_max = lambda_pos * exp(s / (1 - s));
    }

    // ignore extremely small intervals
    if (!(lim.nu_max >= D2 * MINIMUM_RANGE))
        lim.nu_max = 0;
    if (!(lim.lambda_max - lim.lambda_min >= D2 * MINIMUM_RANGE))
        lim.lambda_min = lim.lambda_max = f.lambda;

    lim.order = integr_order(sqrt((lim.lambda_min - D2) / (lim.lambda_max - D2)));
    return lim;
}

/* ------------------------------------------------------------
    Integrand reparameterization

    We have to compute a bunch of integrals like Jr = 1/π int (lo, hi) dtau p(tau) (see
    Eqn. 6 in Binney (2012), for example), but at the integration bounds the integrand
    p(tau) -> 0 like sqrt(tau - lo). For the actions, p has an infinite slope at the
    bounds, which G-L doesn't handle well. It is worse for the frequency and angle
    integrals because they contain integrands with 1/p terms that diverge at the
    integration limits (see Eqn. 15 in Binney (2012), for example). In agama, the change
    of variables goes to a cubic in a new variable "s":

        tau(s) = lo + (hi - lo) s^2 (3 - 2s)

    that has nicer integration properties - mainly that the singularities at the
    integration bounds go away.

    Here, I picked a different reparameterization because I didn't totally follow the
    logic in the cubic CoV in Agama. I instead use a sine:

        tau(s) = lo + (hi - lo) sin^2(π s / 2),    dtau/ds = (hi - lo) (π/2) sin(π s)

    Similar to the cubic function, dtau/ds vanishes at both ends, so near the limits,
    tau - lo ~ s^2 and p ~ s. I think the fixed-order G-L in s should still converge
    quickly. Also the transformation to/from is more straightforward than undoing hte
    cubic map. More precisely, the angles need integrals from the start of each interval
    up to the current point, so we need s(tau). For the sine transformation this is
    closed-form:

        s(tau) = (2/π) atan2(sqrt(tau - lo), sqrt(hi - tau))

    which is well conditioned at both ends. To undo the cubic, Agama has a hard-coded
    initial guess and some Newton iterations. The sine transformation is more expensive
    but I don't think this will be the dominant cost (relative to potential evaluations)
    for most use cases.
*/

// tau(s) = lo + (hi - lo) sin^2(pi s / 2)
double unscale_sin(double lo, double hi, double s, double *duds) {
    const double c = cos(M_PI_2 * s), sn = sin(M_PI_2 * s);
    *duds = (hi - lo) * M_PI * sn * c;  // = (hi - lo) (pi/2) sin(pi s)
    return lo * c * c + hi * sn * sn;   // symmetric form, accurate near both ends
}

double scale_sin(double lo, double hi, double tau) {
    return M_2_PI * atan2(sqrt(fmax(0., tau - lo)), sqrt(fmax(0., hi - tau)));
}

/*
    WARNING: CLAUDE wrote this; I audited.
    Gauss-Legendre integrals over tau in [lo, hi] (scaled s from 0 to s_end) of
     out[0] = p
     out[1] = p^-1 / (tau - Delta^2)
     out[2] = p^-1 / ((tau - Delta^2) tau)
     out[3] = p^-1 / (tau - Delta^2)^2  (minus its singular part on the nu branch)
   where p^2 < 0 or non-finite integrands contribute zero (as in Agama)
*/
void integrate_path(const Fudge &f, double lo, double hi, double s_end, int order,
                    double nu_max, double dFdnu_max, double *out) {
    out[0] = out[1] = out[2] = out[3] = 0;
    if (!(s_end > 0) || hi == lo)
        return;
    const gsl_integration_glfixed_table *table = gl_table(order);
    for (int i = 0; i < order; i++) {
        double s, w, duds;
        gsl_integration_glfixed_point(0., s_end, i, &s, &w, table);
        const double tau = unscale_sin(lo, hi, s, &duds);
        const double tmd = tau - f.D2;
        double F;
        F_tau(f, tau, &F);
        const double p2 = F / (2 * sq(tmd) * tau);
        if (!(p2 > 0))
            continue;
        const double p = sqrt(p2), wt = w * duds;
        double v[4] = {p, 1 / (p * tmd), 1 / (p * tmd * tau), 1 / (p * sq(tmd))};
        if (tmd < 0)  // singular part, integrated analytically by the caller
            v[3] += sqrt(2 * nu_max / dFdnu_max / (tau - nu_max)) / tmd;
        for (int k = 0; k < 4; k++)
            if (std::isfinite(v[k]))
                out[k] += wt * v[k];
    }
}

/*
    This is the main Staeckel Fudge implementation for a single phase-space position w.
*/
void staeckel_fudge_single(CPotential *pot, double t, const double *w, double delta, double *out) {
    // "out" contains the actions (0,1,2), angles (3,4,5), and frequencies (6,7,8)
    for (int k = 0; k < 9; k++)
        out[k] = NAN;

    const double x = w[0], y = w[1], z = w[2], vx = w[3], vy = w[4], vz = w[5];
    const double R = hypot(x, y);
    const double Lz = x * vy - y * vx;
    const double vR = R > 0 ? (x * vx + y * vy) / R : hypot(vx, vy);
    const double vphi = R > 0 ? Lz / R : 0;

    // Jphi = Lz, at least one is easy :)
    out[1] = Lz;

    // heuristic - scale taken from Agama, so should audit
    if (!(delta > 0))
        delta = 1e-4 * hypot(R, z);
    if (!(delta > 0))  // negate to catch nans
        return;

    // Set up our fudge container (mmm...)
    Fudge f;
    f.pot = pot;
    f.t = t;
    f.D2 = delta * delta;
    f.phi = atan2(y, x);
    f.Lz = Lz;

    cyl_to_prolsph(R, z, vR, vz, f.D2, &f.lambda, &f.nu, &f.lambdadot, &f.nudot);

    const double Phi = eval_potential(f, R, z, NULL, NULL);
    const double Phi0 = eval_potential(f, sqrt(f.lambda - f.D2), 0., NULL, NULL);
    f.E = Phi + 0.5 * (vx * vx + vy * vy + vz * vz);
    f.flambda = -f.lambda * Phi0;
    f.fnu = f.flambda + Phi * (f.lambda - fabs(f.nu));
    f.I3 = f.lambda * (Phi - Phi0) +
           0.5 * (sq(z * vphi) + sq(R * vz - z * vR) + sq(vz) * f.D2);

    if (!std::isfinite(f.E + f.I3 + f.Lz) || f.E >= 0)
        return;

    // Get integration limits
    const Limits lim = find_limits(f);
    const bool shell = lim.lambda_min == lim.lambda_max;
    const bool planar = lim.nu_max == 0;

    double dFdnu_max = NAN;
    if (!planar)
        F_tau(f, lim.nu_max, NULL, &dFdnu_max);

    // integrals over the full lambda and nu ranges - this gets us actions and
    // derivatives w.r.t. constants of motion we can compute trivially like E, I3, etc.
    // See: Sanders (2012); Eqns, A1, A4-A9
    double Il[4], In[4];
    integrate_path(f, lim.lambda_min, lim.lambda_max, 1., lim.order, lim.nu_max, dFdnu_max, Il);
    integrate_path(f, 0., lim.nu_max, 1., lim.order, lim.nu_max, dFdnu_max, In);

    if (shell) { // "shell" orbits with no variation in lambda (~radius)
        // the interval collapses to a point: limit of the p^-1 integrals from F''(lambda)
        double d2F;
        F_tau(f, lim.lambda_min, NULL, NULL, &d2F);
        const double tau = lim.lambda_min, tmd = tau - f.D2;
        const double base = 2 * M_PI * sqrt(-tau / d2F) * fabs(tmd);
        Il[0] = 0;
        Il[1] = base / tmd;
        Il[2] = base / (tmd * tau);
        Il[3] = base / sq(tmd);
    }

    // analytic part of the nu integral of p^-1 / (tau - Delta^2)^2 - terms like this
    // appear in the integrals Sanders (2012); Eqns, A1, A4-A9
    const double dmn_max = f.D2 - lim.nu_max;
    const bool has_sing = !planar && f.Lz != 0 && dmn_max > 0;
    const double sing_amp = has_sing ? 2 * sqrt(-2 * lim.nu_max / dFdnu_max / dmn_max) : 0;

    const double Jr = Il[0] / M_PI;
    const double Jz = In[0] / M_PI * 2;  // factor 2: only half the orbit (z >= 0) is integrated
    const double dJrdE = Il[1] / (4 * M_PI);
    const double dJzdE = In[1] / (2 * M_PI);
    const double dJrdI3 = -Il[2] / (4 * M_PI);
    const double dJzdI3 = -In[2] / (2 * M_PI);
    const double dJrdLz = f.Lz == 0 ? 0 : -f.Lz * Il[3] / (4 * M_PI);
    const double dJzdLz =
        f.Lz == 0 ? 0
                  : -f.Lz * (In[3] + sing_amp * atan(sqrt(lim.nu_max / dmn_max))) / (2 * M_PI);

    // invert the Jacobian dJ/d(E, I3, Lz)
    double Omr, Omz, Omphi, dI3dJr, dI3dJz, dI3dJphi;
    const double det = dJrdE * dJzdI3 - dJrdI3 * dJzdE;
    if (planar || det == 0) {
        Omr = 1 / dJrdE;
        Omphi = -dJrdLz / dJrdE;
        Omz = dI3dJr = dI3dJz = dI3dJphi = 0;
    } else {
        Omr = dJzdI3 / det;
        Omz = -dJrdI3 / det;
        Omphi = (dJrdI3 * dJzdLz - dJrdLz * dJzdI3) / det;
        dI3dJr = -dJzdE / det;
        dI3dJz = dJrdE / det;
        dI3dJphi = -(dJrdE * dJzdLz - dJrdLz * dJzdE) / det;
    }

    // derivatives of the generating function (Sanders A10-A12), needed for the angles
    const double sign_l = f.lambdadot >= 0 ? 1 : -1;
    const double sign_n = f.nudot * f.nu >= 0 ? 1 : -1;
    const double yl = shell ? 0 : scale_sin(lim.lambda_min, lim.lambda_max, f.lambda);
    const double yn = planar ? 0 : scale_sin(0., lim.nu_max, fabs(f.nu));
    double Sl[4], Sn[4];
    integrate_path(f, lim.lambda_min, lim.lambda_max, yl, lim.order, lim.nu_max, dFdnu_max, Sl);
    integrate_path(f, 0., lim.nu_max, yn, lim.order, lim.nu_max, dFdnu_max, Sn);

    const double sing_S =
        has_sing ? sing_amp * (atan(sqrt(lim.nu_max / dmn_max)) -
                               atan(sqrt((lim.nu_max - fabs(f.nu)) / dmn_max)))
                 : 0;
    const double dSdE = sign_l * Sl[1] / 4 + sign_n * Sn[1] / 4;
    const double dSdI3 = -sign_l * Sl[2] / 4 - sign_n * Sn[2] / 4;
    const double dSdLz =
        f.Lz == 0 ? 0
                  : f.phi - sign_l * f.Lz * Sl[3] / 4 - sign_n * f.Lz * (Sn[3] + sing_S) / 4;

    // angles (Sanders A3); dLz/dJ = (0, 0, 1)
    const double add_pi_z = (f.nudot < 0 && f.Lz != 0) ? M_PI : 0;
    out[0] = Jr;
    out[2] = Jz;
    out[3] = wrap_angle(dSdE * Omr + dSdI3 * dI3dJr);
    out[4] = wrap_angle(dSdE * Omphi + dSdI3 * dI3dJphi + dSdLz);
    out[5] = wrap_angle(dSdE * Omz + dSdI3 * dI3dJz + add_pi_z);
    out[6] = Omr;
    out[7] = Omphi;
    out[8] = Omz;
}

/* restores the GSL error handler on scope exit, including when a potential errors */
struct GSLErrorHandlerOff {
    gsl_error_handler_t *old;
    GSLErrorHandlerOff() : old(gsl_set_error_handler_off()) {}
    ~GSLErrorHandlerOff() { gsl_set_error_handler(old); }
};

}  // namespace

/*
    Vectorize / loop over phase space positions.

    TODO: ask Lehman if there is anything we can do to vectorize this better?
*/
void staeckel_fudge(CPotential *p, double t, const double *w, const double *delta, size_t n, double *out) {
    GSLErrorHandlerOff guard;
    for (size_t i = 0; i < n; i++)
        staeckel_fudge_single(p, t, &w[6 * i], delta[i], &out[9 * i]);
}

#else

void staeckel_fudge(CPotential *p, double t, const double *w, const double *delta, size_t n, double *out) {
    throw std::runtime_error("The Staeckel fudge requires gala to be built with GSL support.");
}

#endif
