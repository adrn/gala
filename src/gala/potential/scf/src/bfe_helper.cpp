#include <stdlib.h>
#include "extra_compile_macros.h"
#include <math.h>
#include "bfe_helper.h"
#if USE_GSL == 1
#include "gsl/gsl_sf_legendre.h"
#include "gsl/gsl_sf_gegenbauer.h"
#include "gsl/gsl_sf_gamma.h"
#endif

#define SQRT_FOURPI 3.544907701811031

#if USE_GSL == 1
double rho_nl(double s, int n, int l) {
    double RR, Knl;
    Knl = 0.5*n*(n+4*l+3) + (l+1)*(2*l+1);
    RR = Knl/(2*M_PI) * pow(s,l) / (s*pow(1+s,2*l+3)) * gsl_sf_gegenpoly_n(n, 2*l + 1.5, (s-1)/(s+1));
    return SQRT_FOURPI*RR;
}
double rho_nlm(double s, double phi, double X, int n, int l, int m) {
    return rho_nl(s, n, l) * gsl_sf_legendre_sphPlm(l, m, X);// / SQRT_FOURPI;
}

double phi_nl(double s, int n, int l) {
    return -SQRT_FOURPI*pow(s,l) * pow(1+s, -2*l-1) * gsl_sf_gegenpoly_n(n, 2*l+1.5, (s-1)/(s+1));
}
double phi_nlm(double s, double phi, double X, int n, int l, int m) {
    return phi_nl(s, n, l) * gsl_sf_legendre_sphPlm(l, m, X); // / SQRT_FOURPI;
}

void sph_grad_phi_nlm(double s, double phi, double X, int n, int l, int m,
                      int lmax, double *sphgrad) {
    /*
        Spherical components (r, theta, phi) of the gradient of Phi_nl(s) Y_lm,
        excluding the cos(m phi), sin(m phi) factors. The phi component is
        m Y_lm Phi_nl / (s sin(theta)), to be multiplied by d/dphi of those factors.

        On the z axis (sin(theta) = 0), this returns the limit from off the axis.
        At the origin (s = 0, called with X = 1), it returns the s -> 0 limit for
        l >= 1; the l = 0 terms are cusped there, so callers must skip them.
    */
    double A, dYlm_dtheta, mYlm_sintheta, Phinl_s;
    double dPhinl_dr;

    // spherical coord stuff
    double sintheta = sqrt(1-X*X);

    double Phi_nl, Ylm, Plm, Pl1m;
    Phi_nl = phi_nl(s, n, l);

    Ylm = gsl_sf_legendre_sphPlm(l, m, X);

    // Correct: associated Legendre polynomial -- not sphPlm!
    if (m <= l) {
        Plm = gsl_sf_legendre_Plm(l, m, X);
    } else {
        Plm = 0.;
    }

    // copied out of Mathematica
    if (n == 0) {
        dPhinl_dr = SQRT_FOURPI*pow(s,-1 + l)*pow(1 + s,-3 - 2*l)*(1 + s)*(l*(-1 + s) + s);
    } else {
        dPhinl_dr = (SQRT_FOURPI*pow(s,-1 + l)*pow(1 + s,-3 - 2*l)*
                      (-2*(3 + 4*l)*s*gsl_sf_gegenpoly_n(-1 + n, 2.5 + 2*l, (-1 + s)/(1 + s)) +
                      (1 + s)*(l*(-1 + s) + s)*gsl_sf_gegenpoly_n(n, 1.5 + 2*l, (-1 + s)/(1 + s))));
    }

    // Phi_nl / s, which at the origin -> dPhi_nl/ds for l = 1 and 0 for l >= 2
    if (s == 0) {
        Phinl_s = (l == 1) ? dPhinl_dr : 0.;
    } else {
        Phinl_s = Phi_nl / s;
    }
    dPhinl_dr *= Ylm;

    if (l==0) {
        dYlm_dtheta = 0.;
        mYlm_sintheta = 0.;
    } else {
        if (l == m) {
            A = sqrt(2*l+1) / SQRT_FOURPI * sqrt(1. / gsl_sf_gamma(l+m+1.));
        } else {
            A = sqrt(2*l+1) / SQRT_FOURPI * sqrt(gsl_sf_gamma(l-m+1.) / gsl_sf_gamma(l+m+1.));
        }

        if (sintheta == 0) {
            // On the z axis (X = +/-1), only m = 1 terms have nonzero angular
            // derivatives. With the Condon-Shortley phase (as in GSL), the limits are
            // dP_l^1/dtheta -> -l(l+1)/2 X^l and P_l^1 / sin(theta) -> -l(l+1)/2 X^(l+1)
            if (m == 1) {
                dYlm_dtheta = -0.5 * A * l*(l+1) * pow(X, l);
                mYlm_sintheta = -0.5 * A * l*(l+1) * pow(X, l+1);
            } else {
                dYlm_dtheta = 0.;
                mYlm_sintheta = 0.;
            }
        } else {
            // Correct: associated Legendre polynomial -- not sphPlm!
            if (m <= (l-1)) {
                Pl1m = gsl_sf_legendre_Plm(l-1, m, X);
            } else {
                Pl1m = 0.;
            }
            dYlm_dtheta = A / sintheta * (l*X*Plm - (l+m)*Pl1m);
            mYlm_sintheta = m * Ylm / sintheta;
        }
    }

    sphgrad[0] = dPhinl_dr;
    sphgrad[1] = dYlm_dtheta * Phinl_s;
    sphgrad[2] = mYlm_sintheta * Phinl_s;
}
#endif
