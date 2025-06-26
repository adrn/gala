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
    double A, dYlm_dtheta;
    double dPhinl_dr, dPhi_dphi, dPhi_dtheta;

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
    dPhinl_dr *= Ylm;

    if (l==0) {
        dYlm_dtheta = 0.;
    } else {
        // Correct: associated Legendre polynomial -- not sphPlm!
        if (m <= (l-1)) {
            Pl1m = gsl_sf_legendre_Plm(l-1, m, X);
        } else {
            Pl1m = 0.;
        }

        if (l == m) {
            A = sqrt(2*l+1) / SQRT_FOURPI * sqrt(1. / gsl_sf_gamma(l+m+1.));
        } else {
            A = sqrt(2*l+1) / SQRT_FOURPI * sqrt(gsl_sf_gamma(l-m+1.) / gsl_sf_gamma(l+m+1.));
        }
        dYlm_dtheta = A / sintheta * (l*X*Plm - (l+m)*Pl1m);
    }
    dPhi_dtheta = dYlm_dtheta * Phi_nl / s;

    if (m == 0) {
        dPhi_dphi = 0.;
    } else {
        dPhi_dphi = m;
    }
    dPhi_dphi *= Ylm * Phi_nl;

    sphgrad[0] = dPhinl_dr;
    sphgrad[1] = dPhi_dtheta;
    sphgrad[2] = dPhi_dphi;
}
#endif
