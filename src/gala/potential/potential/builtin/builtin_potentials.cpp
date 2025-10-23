#include <math.h>
#include <string.h>
#include <stdio.h>
#include "extra_compile_macros.h"
#include "src/vectorization.h"
#include "potential_helpers.h"

#if USE_GSL == 1
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_math.h>
#endif

double nan_density(double t, double *pars, double *q, int n_dim, void *state) { return NAN; }
double nan_value(double t, double *pars, double *q, int n_dim, void *state) { return NAN; }
void nan_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {}
void nan_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {}

double null_density(double t, double *pars, double *q, int n_dim, void *state) { return 0; }
double null_value(double t, double *pars, double *q, int n_dim, void *state) { return 0; }
void null_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state){}
void null_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {}

/* Note: many Hessians generated with sympy in
    gala-notebooks/Make-all-Hessians.ipynb
*/

/* ---------------------------------------------------------------------------
    Henon-Heiles potential
*/
double henon_heiles_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  no parameters... */
    return 0.5 * (q[0]*q[0] + q[1]*q[1] + 2*q[0]*q[0]*q[1] - 2/3.*q[1]*q[1]*q[1]);
}

void henon_heiles_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  no parameters... */
    grad[0] = grad[0] + q[0] + 2*q[0]*q[1];
    grad[1] = grad[1] + q[1] + q[0]*q[0] - q[1]*q[1];
}

void henon_heiles_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  no parameters... */
    double x = q[0];
    double y = q[1];

    double tmp_0 = 2.0 * y;
    double tmp_1 = 2.0 * x;

    hess[0] = hess[0] + tmp_0 + 1.0;
    hess[1] = hess[1] + tmp_1;
    hess[2] = hess[2] + tmp_1;
    hess[3] = hess[3] + 1.0 - tmp_0;
}

/* ---------------------------------------------------------------------------
    Kepler potential
*/
double kepler_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    return -pars[0] * pars[1] / norm3(q);
}

void kepler_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    const double fac = pars[0] * pars[1] / pow(norm3(q), 3);

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double kepler_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    if (norm3_sq(q) == 0.) {
        return INFINITY;
    } else {
        return 0.;
    }
}

void kepler_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    double G = pars[0];
    double m = pars[1];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(y, 2);
    double tmp_2 = pow(z, 2);
    double tmp_3 = tmp_0 + tmp_1 + tmp_2;
    double tmp_4 = G*m;
    double tmp_5 = tmp_4/pow(tmp_3, 3.0/2.0);
    double tmp_6 = 3*tmp_4/pow(tmp_3, 5.0/2.0);
    double tmp_7 = tmp_6*x;
    double tmp_8 = -tmp_7*y;
    double tmp_9 = -tmp_7*z;
    double tmp_10 = -tmp_6*y*z;

    hess[0] = hess[0] + -tmp_0*tmp_6 + tmp_5;
    hess[1] = hess[1] + tmp_8;
    hess[2] = hess[2] + tmp_9;
    hess[3] = hess[3] + tmp_8;
    hess[4] = hess[4] + -tmp_1*tmp_6 + tmp_5;
    hess[5] = hess[5] + tmp_10;
    hess[6] = hess[6] + tmp_9;
    hess[7] = hess[7] + tmp_10;
    hess[8] = hess[8] + -tmp_2*tmp_6 + tmp_5;
}

/* ---------------------------------------------------------------------------
    Isochrone potential
*/
double isochrone_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    const double r2 = norm3_sq(q);
    return -pars[0] * pars[1] / (sqrt(r2 + pars[2]*pars[2]) + pars[2]);
}

void isochrone_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    const double sqrt_r2_b2 = sqrt(norm3_sq(q) + pars[2]*pars[2]);
    const double denom = sqrt_r2_b2 * (sqrt_r2_b2 + pars[2])*(sqrt_r2_b2 + pars[2]);
    const double fac = pars[0] * pars[1] / denom;

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double isochrone_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    const double b = pars[2];
    const double r2 = norm3_sq(q);
    const double a = sqrt(b*b + r2);

    return pars[1] * (3*(b+a)*a*a - r2*(b+3*a)) / (4*M_PI*pow(b+a,3)*a*a*a);
}

void isochrone_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double G = pars[0];
    double m = pars[1];
    double b = pars[2];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(y, 2);
    double tmp_2 = pow(z, 2);
    double tmp_3 = pow(b, 2) + tmp_0 + tmp_1 + tmp_2;
    double tmp_4 = sqrt(tmp_3);
    double tmp_5 = b + tmp_4;
    double tmp_6 = G*m;
    double tmp_7 = tmp_6/pow(tmp_5, 2);
    double tmp_8 = tmp_7/tmp_4;
    double tmp_9 = 2*tmp_6/(tmp_3*pow(tmp_5, 3));
    double tmp_10 = tmp_7/pow(tmp_3, 3.0/2.0);
    double tmp_11 = tmp_9*x;
    double tmp_12 = tmp_10*x;
    double tmp_13 = -tmp_11*y - tmp_12*y;
    double tmp_14 = -tmp_11*z - tmp_12*z;
    double tmp_15 = y*z;
    double tmp_16 = -tmp_10*tmp_15 - tmp_15*tmp_9;

    hess[0] = hess[0] + -tmp_0*tmp_10 - tmp_0*tmp_9 + tmp_8;
    hess[1] = hess[1] + tmp_13;
    hess[2] = hess[2] + tmp_14;
    hess[3] = hess[3] + tmp_13;
    hess[4] = hess[4] + -tmp_1*tmp_10 - tmp_1*tmp_9 + tmp_8;
    hess[5] = hess[5] + tmp_16;
    hess[6] = hess[6] + tmp_14;
    hess[7] = hess[7] + tmp_16;
    hess[8] = hess[8] + -tmp_10*tmp_2 - tmp_2*tmp_9 + tmp_8;

}

/* ---------------------------------------------------------------------------
    Hernquist sphere
*/
double hernquist_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    const double r = norm3(q);
    return -pars[0] * pars[1] / (r + pars[2]);
}

void hernquist_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    const double r = norm3(q);
    const double fac = pars[0] * pars[1] / ((r + pars[2]) * (r + pars[2]) * r);

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double hernquist_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    const double r = norm3(q);
    const double rho0 = pars[1]/(2*M_PI*pars[2]*pars[2]*pars[2]);
    return rho0 / ((r/pars[2]) * pow(1+r/pars[2],3));
}

void hernquist_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double G = pars[0];
    double m = pars[1];
    double c = pars[2];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(y, 2);
    double tmp_2 = pow(z, 2);
    double tmp_3 = tmp_0 + tmp_1 + tmp_2;
    double tmp_4 = sqrt(tmp_3);
    double tmp_5 = c + tmp_4;
    double tmp_6 = G*m;
    double tmp_7 = tmp_6/pow(tmp_5, 2);
    double tmp_8 = tmp_7/tmp_4;
    double tmp_9 = 2*tmp_6/(tmp_3*pow(tmp_5, 3));
    double tmp_10 = tmp_7/pow(tmp_3, 3.0/2.0);
    double tmp_11 = tmp_9*x;
    double tmp_12 = tmp_10*x;
    double tmp_13 = -tmp_11*y - tmp_12*y;
    double tmp_14 = -tmp_11*z - tmp_12*z;
    double tmp_15 = y*z;
    double tmp_16 = -tmp_10*tmp_15 - tmp_15*tmp_9;

    hess[0] = hess[0] + -tmp_0*tmp_10 - tmp_0*tmp_9 + tmp_8;
    hess[1] = hess[1] + tmp_13;
    hess[2] = hess[2] + tmp_14;
    hess[3] = hess[3] + tmp_13;
    hess[4] = hess[4] + -tmp_1*tmp_10 - tmp_1*tmp_9 + tmp_8;
    hess[5] = hess[5] + tmp_16;
    hess[6] = hess[6] + tmp_14;
    hess[7] = hess[7] + tmp_16;
    hess[8] = hess[8] + -tmp_10*tmp_2 - tmp_2*tmp_9 + tmp_8;
}


/* ---------------------------------------------------------------------------
    Plummer sphere
*/
double plummer_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    const double r2 = norm3_sq(q);
    return -pars[0]*pars[1] / sqrt(r2 + pars[2]*pars[2]);
}

void plummer_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    const double R2b = norm3_sq(q) + pars[2]*pars[2];
    const double fac = pars[0] * pars[1] / sqrt(R2b) / R2b;

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double plummer_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    const double r2 = norm3_sq(q);
    return 3*pars[1] / (4*M_PI*pars[2]*pars[2]*pars[2]) * pow(1 + r2/(pars[2]*pars[2]), -2.5);
}

void plummer_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double G = pars[0];
    double m = pars[1];
    double b = pars[2];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(y, 2);
    double tmp_2 = pow(z, 2);
    double tmp_3 = pow(b, 2) + tmp_0 + tmp_1 + tmp_2;
    double tmp_4 = G*m;
    double tmp_5 = tmp_4/pow(tmp_3, 3.0/2.0);
    double tmp_6 = 3*tmp_4/pow(tmp_3, 5.0/2.0);
    double tmp_7 = tmp_6*x;
    double tmp_8 = -tmp_7*y;
    double tmp_9 = -tmp_7*z;
    double tmp_10 = -tmp_6*y*z;

    hess[0] = hess[0] + -tmp_0*tmp_6 + tmp_5;
    hess[1] = hess[1] + tmp_8;
    hess[2] = hess[2] + tmp_9;
    hess[3] = hess[3] + tmp_8;
    hess[4] = hess[4] + -tmp_1*tmp_6 + tmp_5;
    hess[5] = hess[5] + tmp_10;
    hess[6] = hess[6] + tmp_9;
    hess[7] = hess[7] + tmp_10;
    hess[8] = hess[8] + -tmp_2*tmp_6 + tmp_5;
}

/* ---------------------------------------------------------------------------
    Jaffe sphere
*/
double jaffe_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    const double r = norm3(q);
    return -pars[0] * pars[1] / pars[2] * log(1 + pars[2] / r);
}

void jaffe_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state){
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    const double r = norm3(q);
    const double fac = pars[0] * pars[1] / pars[2] * (pars[2] / (r * (pars[2] + r))) / r;

    grad[0] = grad[0] + fac * q[0];
    grad[1] = grad[1] + fac * q[1];
    grad[2] = grad[2] + fac * q[2];
}

double jaffe_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    const double r = norm3(q);
    const double rho0 = pars[1] / (4*M_PI*pars[2]*pars[2]*pars[2]);
    return rho0 / (pow(r/pars[2],2) * pow(1+r/pars[2],2));
}

void jaffe_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double G = pars[0];
    double m = pars[1];
    double c = pars[2];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(y, 2);
    double tmp_2 = pow(z, 2);
    double tmp_3 = tmp_0 + tmp_1 + tmp_2;
    double tmp_4 = 1.0/tmp_3;
    double tmp_5 = sqrt(tmp_3);
    double tmp_6 = c + tmp_5;
    double tmp_7 = pow(tmp_6, -2);
    double tmp_8 = tmp_7*x;
    double tmp_9 = 1.0/tmp_5;
    double tmp_10 = 1.0/tmp_6;
    double tmp_11 = tmp_10*tmp_9;
    double tmp_12 = G*m/c;
    double tmp_13 = tmp_12*(tmp_11*x - tmp_8);
    double tmp_14 = tmp_13*tmp_4;
    double tmp_15 = pow(tmp_3, -3.0/2.0);
    double tmp_16 = tmp_13*tmp_15*tmp_6;
    double tmp_17 = tmp_10*tmp_15;
    double tmp_18 = tmp_4*tmp_7;
    double tmp_19 = 2*tmp_9/pow(tmp_6, 3);
    double tmp_20 = tmp_11 - tmp_7;
    double tmp_21 = tmp_12*tmp_6;
    double tmp_22 = tmp_21*tmp_9;
    double tmp_23 = tmp_19*x;
    double tmp_24 = tmp_4*tmp_8;
    double tmp_25 = tmp_17*x;
    double tmp_26 = tmp_14*y - tmp_16*y + tmp_22*(tmp_23*y - tmp_24*y - tmp_25*y);
    double tmp_27 = tmp_4*z;
    double tmp_28 = tmp_13*tmp_27 - tmp_16*z + tmp_22*(tmp_23*z - tmp_24*z - tmp_25*z);
    double tmp_29 = tmp_7*y;
    double tmp_30 = tmp_11*y - tmp_29;
    double tmp_31 = tmp_12*tmp_30;
    double tmp_32 = tmp_15*tmp_21;
    double tmp_33 = tmp_30*tmp_32;
    double tmp_34 = y*z;
    double tmp_35 = tmp_22*(-tmp_17*tmp_34 + tmp_19*tmp_34 - tmp_27*tmp_29) + tmp_27*tmp_31 - tmp_33*z;
    double tmp_36 = tmp_11*z - tmp_7*z;

    hess[0] = hess[0] + tmp_14*x - tmp_16*x + tmp_22*(-tmp_0*tmp_17 - tmp_0*tmp_18 + tmp_0*tmp_19 + tmp_20);
    hess[1] = hess[1] + tmp_26;
    hess[2] = hess[2] + tmp_28;
    hess[3] = hess[3] + tmp_26;
    hess[4] = hess[4] + tmp_22*(-tmp_1*tmp_17 - tmp_1*tmp_18 + tmp_1*tmp_19 + tmp_20) + tmp_31*tmp_4*y - tmp_33*y;
    hess[5] = hess[5] + tmp_35;
    hess[6] = hess[6] + tmp_28;
    hess[7] = hess[7] + tmp_35;
    hess[8] = hess[8] + tmp_12*tmp_27*tmp_36 + tmp_22*(-tmp_17*tmp_2 - tmp_18*tmp_2 + tmp_19*tmp_2 + tmp_20) - tmp_32*tmp_36*z;
}

/* ---------------------------------------------------------------------------
    Power-law potential with exponential cutoff
*/
#if USE_GSL == 1

double safe_gamma_inc(double a, double x) {
    int N, m, n;
    double A = 1.;
    double B = 0.;
    double tmp;

    if (a > 0) {
        return gsl_sf_gamma_inc_P(a, x) * gsl_sf_gamma(a);;
    } else {
        N = (int) ceil(-a);

        for (n=0; n < N; n++) {
            A = A * (a + n);

            tmp = 1.;
            for (m=N-1; m > n; m--) {
                tmp = tmp * (a + m);
            }
            B = B + pow(x, a+n) * exp(-x) * tmp;
        }
        return (B + gsl_sf_gamma_inc_P(a + N, x) * gsl_sf_gamma(a + N)) / A;
    }
}

double powerlawcutoff_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 - m (total mass)
            2 - a (power-law index)
            3 - c (cutoff radius)
    */
    const double G = pars[0];
    const double m = pars[1];
    const double alpha = pars[2];
    const double r_c = pars[3];
    const double r = norm3(q);

    if (r == 0.) {
        return -INFINITY;
    } else {
        const double tmp_0 = alpha / 2.0;
        const double tmp_1 = -tmp_0;
        const double tmp_2 = tmp_1 + 1.5;
        const double tmp_3 = r * r;
        const double tmp_4 = tmp_3 / pow(r_c, 2);
        const double tmp_5 = G*m;
        const double tmp_6 = tmp_5*safe_gamma_inc(tmp_2, tmp_4)/(sqrt(tmp_3)*tgamma(tmp_1 + 2.5));

        // Original potential
        double phi_r = tmp_0*tmp_6 - 3.0/2.0*tmp_6 + tmp_5*safe_gamma_inc(tmp_1 + 1, tmp_4)/(r_c*tgamma(tmp_2));

        // Subtract asymptotic value to enforce Φ(∞) = 0
        double phi_infinity = 0.0;
        if (tmp_2 > 0) {  // alpha < 3
            phi_infinity = tmp_5 * tgamma(tmp_1 + 1) / (r_c * tgamma(tmp_2));
        }

        return phi_r - phi_infinity;
    }
}

double powerlawcutoff_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 - m (total mass)
            2 - a (power-law index)
            3 - c (cutoff radius)
    */
    const double r = norm3(q);
    const double A = pars[1] / (2*M_PI) * pow(pars[3], pars[2] - 3) / gsl_sf_gamma(0.5 * (3 - pars[2]));
    return A * pow(r, -pars[2]) * exp(-r*r / (pars[3]*pars[3]));
}

void powerlawcutoff_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 - m (total mass)
            2 - a (power-law index)
            3 - c (cutoff radius)
    */
    const double r = norm3(q);
    const double dPhi_dr = (pars[0] * pars[1] / (r*r * r) *
        gsl_sf_gamma_inc_P(0.5 * (3-pars[2]), r*r/(pars[3]*pars[3]))); // / gsl_sf_gamma(0.5 * (3-pars[2])));

    grad[0] = grad[0] + dPhi_dr * q[0];
    grad[1] = grad[1] + dPhi_dr * q[1];
    grad[2] = grad[2] + dPhi_dr * q[2];
}

void powerlawcutoff_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - alpha (exponent)
            - r_c (cutoff radius)
    */
    double G = pars[0];
    double m = pars[1];
    double alpha = pars[2];
    double r_c = pars[3];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(y, 2);
    double tmp_2 = pow(z, 2);
    double tmp_3 = tmp_0 + tmp_1 + tmp_2;
    double tmp_4 = (1.0/2.0)*alpha;
    double tmp_5 = -tmp_4;
    double tmp_6 = tmp_5 + 1.5;
    double tmp_7 = pow(r_c, -2);
    double tmp_8 = tmp_3*tmp_7;
    double tmp_9 = G*m;
    double tmp_10 = tmp_9/tgamma(tmp_5 + 2.5);
    double tmp_11 = tmp_10*safe_gamma_inc(tmp_6, tmp_8);
    double tmp_12 = tmp_11/pow(tmp_3, 5.0/2.0);
    double tmp_13 = (9.0/2.0)*tmp_12;
    double tmp_14 = exp(-tmp_8);
    double tmp_15 = tmp_0*tmp_14;
    double tmp_16 = pow(tmp_8, -tmp_4)*tmp_9/tgamma(tmp_6);
    double tmp_17 = 4*tmp_16/pow(r_c, 5);
    double tmp_18 = alpha*tmp_0;
    double tmp_19 = (3.0/2.0)*tmp_12;
    double tmp_20 = 6*tmp_15;
    double tmp_21 = pow(r_c, -4);
    double tmp_22 = tmp_5 + 0.5;
    double tmp_23 = pow(tmp_8, tmp_22);
    double tmp_24 = tmp_10*tmp_23/sqrt(tmp_3);
    double tmp_25 = tmp_21*tmp_24;
    double tmp_26 = pow(tmp_3, -3.0/2.0);
    double tmp_27 = tmp_10*tmp_23*tmp_26*tmp_7;
    double tmp_28 = tmp_20*tmp_27;
    double tmp_29 = 2*tmp_14;
    double tmp_30 = tmp_18*tmp_29;
    double tmp_31 = tmp_16*tmp_29/pow(r_c, 3);
    double tmp_32 = tmp_31/tmp_3;
    double tmp_33 = tmp_27*tmp_30;
    double tmp_34 = tmp_11*tmp_26;
    double tmp_35 = tmp_14*tmp_24;
    double tmp_36 = tmp_35*tmp_7;
    double tmp_37 = alpha*tmp_36 + tmp_31 - tmp_34*tmp_4 + (3.0/2.0)*tmp_34 - 3*tmp_36;
    double tmp_38 = tmp_13*x;
    double tmp_39 = alpha*tmp_19;
    double tmp_40 = x*y;
    double tmp_41 = tmp_14*tmp_17;
    double tmp_42 = alpha*tmp_40;
    double tmp_43 = tmp_21*tmp_35;
    double tmp_44 = 6*tmp_40;
    double tmp_45 = tmp_14*tmp_27;
    double tmp_46 = tmp_44*tmp_45;
    double tmp_47 = tmp_29*tmp_42;
    double tmp_48 = tmp_27*tmp_47;
    double tmp_49 = -tmp_22*tmp_46 + tmp_22*tmp_48 - tmp_25*tmp_47 - tmp_32*tmp_42 - tmp_38*y + tmp_39*tmp_40 - tmp_40*tmp_41 + tmp_43*tmp_44 + tmp_46 - tmp_48;
    double tmp_50 = x*z;
    double tmp_51 = alpha*tmp_50;
    double tmp_52 = 6*tmp_50;
    double tmp_53 = tmp_45*tmp_52;
    double tmp_54 = tmp_29*tmp_51;
    double tmp_55 = tmp_27*tmp_54;
    double tmp_56 = -tmp_22*tmp_53 + tmp_22*tmp_55 - tmp_25*tmp_54 - tmp_32*tmp_51 - tmp_38*z + tmp_39*tmp_50 - tmp_41*tmp_50 + tmp_43*tmp_52 + tmp_53 - tmp_55;
    double tmp_57 = 6*tmp_1;
    double tmp_58 = tmp_45*tmp_57;
    double tmp_59 = alpha*tmp_1;
    double tmp_60 = tmp_29*tmp_59;
    double tmp_61 = tmp_27*tmp_60;
    double tmp_62 = y*z;
    double tmp_63 = alpha*tmp_62;
    double tmp_64 = 6*tmp_62;
    double tmp_65 = tmp_45*tmp_64;
    double tmp_66 = tmp_29*tmp_63;
    double tmp_67 = tmp_27*tmp_66;
    double tmp_68 = -tmp_13*tmp_62 - tmp_22*tmp_65 + tmp_22*tmp_67 - tmp_25*tmp_66 - tmp_32*tmp_63 + tmp_39*tmp_62 - tmp_41*tmp_62 + tmp_43*tmp_64 + tmp_65 - tmp_67;
    double tmp_69 = 6*tmp_2;
    double tmp_70 = tmp_45*tmp_69;
    double tmp_71 = alpha*tmp_2;
    double tmp_72 = tmp_29*tmp_71;
    double tmp_73 = tmp_27*tmp_72;

    hess[0] = hess[0] + -tmp_0*tmp_13 - tmp_15*tmp_17 + tmp_18*tmp_19 - tmp_18*tmp_32 + tmp_20*tmp_25 - tmp_22*tmp_28 + tmp_22*tmp_33 - tmp_25*tmp_30 + tmp_28 - tmp_33 + tmp_37;
    hess[1] = hess[1] + tmp_49;
    hess[2] = hess[2] + tmp_56;
    hess[3] = hess[3] + tmp_49;
    hess[4] = hess[4] + -tmp_1*tmp_13 + tmp_1*tmp_39 - tmp_1*tmp_41 - tmp_22*tmp_58 + tmp_22*tmp_61 - tmp_25*tmp_60 - tmp_32*tmp_59 + tmp_37 + tmp_43*tmp_57 + tmp_58 - tmp_61;
    hess[5] = hess[5] + tmp_68;
    hess[6] = hess[6] + tmp_56;
    hess[7] = hess[7] + tmp_68;
    hess[8] = hess[8] + -tmp_13*tmp_2 + tmp_2*tmp_39 - tmp_2*tmp_41 - tmp_22*tmp_70 + tmp_22*tmp_73 - tmp_25*tmp_72 - tmp_32*tmp_71 + tmp_37 + tmp_43*tmp_69 + tmp_70 - tmp_73;
}

#endif

/* ---------------------------------------------------------------------------
    Stone-Ostriker potential from Stone & Ostriker (2015)
*/
double stone_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - M (total mass)
            - r_c (core radius)
            - r_h (halo radius)
    */
    const double r = norm3(q);
    const double u_c = r / pars[2];
    const double u_h = r / pars[3];

    const double fac = 2*pars[0]*pars[1] / M_PI / (pars[3] - pars[2]);

    if (r == 0) {
        return -fac * 0.5 * log(pars[3]*pars[3] / (pars[2] * pars[2]));
    } else {
        return -fac * (
            atan(u_h)/u_h - atan(u_c)/u_c +
            0.5*log((r*r + pars[3]*pars[3])/(r*r + pars[2]*pars[2]))
        );
    }

}

void stone_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - M (total mass)
            - r_c (core radius)
            - r_h (halo radius)
    */

    const double r = norm3(q);
    const double u_c = r / pars[2];
    const double u_h = r / pars[3];

    const double fac = 2*pars[0]*pars[1] / (M_PI*r*r * r) / (pars[2] - pars[3]);  // order flipped from value
    const double dphi_dr = fac * (pars[2]*atan(u_c) - pars[3]*atan(u_h));

    grad[0] = grad[0] + dphi_dr*q[0];
    grad[1] = grad[1] + dphi_dr*q[1];
    grad[2] = grad[2] + dphi_dr*q[2];
}

double stone_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - M (total mass)
            - r_c (core radius)
            - r_h (halo radius)
    */
    const double r = norm3(q);
    const double rho = pars[1] * (pars[2] + pars[3]) / (2*M_PI*M_PI*pars[2]*pars[2]*pars[3]*pars[3]);
    const double u_c = r / pars[2];
    const double u_t = r / pars[3];

    return rho / ((1 + u_c*u_c)*(1 + u_t*u_t));
}

void stone_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_c (core radius)
            - r_h (halo radius)
    */
    double G = pars[0];
    double m = pars[1];
    double r_c = pars[2];
    double r_h = pars[3];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(r_h, 2);
    double tmp_1 = 1.0/tmp_0;
    double tmp_2 = pow(x, 2);
    double tmp_3 = pow(y, 2);
    double tmp_4 = pow(z, 2);
    double tmp_5 = tmp_2 + tmp_3 + tmp_4;
    double tmp_6 = tmp_1*tmp_5 + 1;
    double tmp_7 = 1.0/tmp_6;
    double tmp_8 = 3/pow(tmp_5, 2);
    double tmp_9 = tmp_2*tmp_8;
    double tmp_10 = pow(r_c, 2);
    double tmp_11 = 1.0/tmp_10;
    double tmp_12 = tmp_11*tmp_5 + 1;
    double tmp_13 = 1.0/tmp_12;
    double tmp_14 = tmp_10 + tmp_5;
    double tmp_15 = pow(tmp_14, -2);
    double tmp_16 = 8*tmp_15;
    double tmp_17 = tmp_0 + tmp_5;
    double tmp_18 = 8*tmp_17/pow(tmp_14, 3);
    double tmp_19 = 2/tmp_14;
    double tmp_20 = 2*tmp_15*tmp_17;
    double tmp_21 = tmp_19 - tmp_20;
    double tmp_22 = 1.0/tmp_17;
    double tmp_23 = 0.5*tmp_14*tmp_22;
    double tmp_24 = tmp_19*x - tmp_20*x;
    double tmp_25 = 1.0*tmp_22;
    double tmp_26 = tmp_24*tmp_25;
    double tmp_27 = sqrt(tmp_5);
    double tmp_28 = r_c*atan(tmp_27/r_c);
    double tmp_29 = 3/pow(tmp_5, 5.0/2.0);
    double tmp_30 = tmp_2*tmp_29;
    double tmp_31 = 1.0/tmp_5;
    double tmp_32 = 2*tmp_31;
    double tmp_33 = tmp_2*tmp_32;
    double tmp_34 = tmp_1/pow(tmp_6, 2);
    double tmp_35 = tmp_11/pow(tmp_12, 2);
    double tmp_36 = r_h*atan(tmp_27/r_h);
    double tmp_37 = 1.0*tmp_14/pow(tmp_17, 2);
    double tmp_38 = tmp_24*tmp_37;
    double tmp_39 = pow(tmp_5, -3.0/2.0);
    double tmp_40 = -tmp_13*tmp_31 + tmp_28*tmp_39 + tmp_31*tmp_7 - tmp_36*tmp_39;
    double tmp_41 = 2*G*m/(-3.1415926535897931*r_c + 3.1415926535897931*r_h);
    double tmp_42 = x*y;
    double tmp_43 = tmp_42*tmp_8;
    double tmp_44 = tmp_29*tmp_42;
    double tmp_45 = tmp_32*tmp_42;
    double tmp_46 = tmp_16*x;
    double tmp_47 = -tmp_41*(tmp_13*tmp_43 + tmp_23*(tmp_18*tmp_42 - tmp_46*y) + tmp_26*y - tmp_28*tmp_44 - tmp_34*tmp_45 + tmp_35*tmp_45 + tmp_36*tmp_44 - tmp_38*y - tmp_43*tmp_7);
    double tmp_48 = x*z;
    double tmp_49 = tmp_48*tmp_8;
    double tmp_50 = tmp_29*tmp_48;
    double tmp_51 = tmp_32*tmp_48;
    double tmp_52 = -tmp_41*(tmp_13*tmp_49 + tmp_23*(tmp_18*tmp_48 - tmp_46*z) + tmp_26*z - tmp_28*tmp_50 - tmp_34*tmp_51 + tmp_35*tmp_51 + tmp_36*tmp_50 - tmp_38*z - tmp_49*tmp_7);
    double tmp_53 = tmp_3*tmp_8;
    double tmp_54 = tmp_19*y - tmp_20*y;
    double tmp_55 = tmp_25*tmp_54;
    double tmp_56 = tmp_29*tmp_3;
    double tmp_57 = tmp_3*tmp_32;
    double tmp_58 = tmp_37*tmp_54;
    double tmp_59 = y*z;
    double tmp_60 = tmp_59*tmp_8;
    double tmp_61 = tmp_29*tmp_59;
    double tmp_62 = tmp_32*tmp_59;
    double tmp_63 = -tmp_41*(tmp_13*tmp_60 + tmp_23*(-tmp_16*tmp_59 + tmp_18*tmp_59) - tmp_28*tmp_61 - tmp_34*tmp_62 + tmp_35*tmp_62 + tmp_36*tmp_61 + tmp_55*z - tmp_58*z - tmp_60*tmp_7);
    double tmp_64 = tmp_4*tmp_8;
    double tmp_65 = z*(tmp_19*z - tmp_20*z);
    double tmp_66 = tmp_29*tmp_4;
    double tmp_67 = tmp_32*tmp_4;

    hess[0] = hess[0] + -tmp_41*(tmp_13*tmp_9 + tmp_23*(-tmp_16*tmp_2 + tmp_18*tmp_2 + tmp_21) + tmp_26*x - tmp_28*tmp_30 + tmp_30*tmp_36 - tmp_33*tmp_34 + tmp_33*tmp_35 - tmp_38*x + tmp_40 - tmp_7*tmp_9);
    hess[1] = hess[1] + tmp_47;
    hess[2] = hess[2] + tmp_52;
    hess[3] = hess[3] + tmp_47;
    hess[4] = hess[4] + -tmp_41*(tmp_13*tmp_53 + tmp_23*(-tmp_16*tmp_3 + tmp_18*tmp_3 + tmp_21) - tmp_28*tmp_56 - tmp_34*tmp_57 + tmp_35*tmp_57 + tmp_36*tmp_56 + tmp_40 - tmp_53*tmp_7 + tmp_55*y - tmp_58*y);
    hess[5] = hess[5] + tmp_63;
    hess[6] = hess[6] + tmp_52;
    hess[7] = hess[7] + tmp_63;
    hess[8] = hess[8] + -tmp_41*(tmp_13*tmp_64 + tmp_23*(-tmp_16*tmp_4 + tmp_18*tmp_4 + tmp_21) + tmp_25*tmp_65 - tmp_28*tmp_66 - tmp_34*tmp_67 + tmp_35*tmp_67 + tmp_36*tmp_66 - tmp_37*tmp_65 + tmp_40 - tmp_64*tmp_7);
}

/* ---------------------------------------------------------------------------
    Spherical NFW
*/
double sphericalnfw_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (scale radius)
    */
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    const double v_h2 = -pars[0] * pars[1] / pars[2];
    const double u = norm3(q) / pars[2];
    if (u == 0) {
        return v_h2;
    } else {
        return v_h2 * log(1 + u) / u;
    }
}

void sphericalnfw_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (scale radius)
    */
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    const double v_h2 = pars[0] * pars[1] / pars[2];

    const double u = norm3(q) / pars[2];
    const double fac = v_h2 / (u*u*u) / (pars[2]*pars[2]) * (log(1+u) - u/(1+u));

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double sphericalnfw_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (scale radius)
    */
    // double v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    const double v_h2 = pars[0] * pars[1] / pars[2];
    const double r = norm3(q);

    const double rho0 = v_h2 / (4*M_PI*pars[0]*pars[2]*pars[2]);
    return rho0 / ((r/pars[2]) * pow(1+r/pars[2],2));
}

void sphericalnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (scale radius)
    */
    double G = pars[0];
    double m = pars[1];
    double r_s = pars[2];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(y, 2);
    double tmp_2 = pow(z, 2);
    double tmp_3 = tmp_0 + tmp_1 + tmp_2;
    double tmp_4 = pow(tmp_3, 7);
    double tmp_5 = 3*tmp_0;
    double tmp_6 = sqrt(tmp_3);
    double tmp_7 = r_s + tmp_6;
    double tmp_8 = pow(tmp_3, 13.0/2.0)*tmp_7;
    double tmp_9 = pow(tmp_7, 2);
    double tmp_10 = 1.0/r_s;
    double tmp_11 = tmp_9*log(tmp_10*tmp_7);
    double tmp_12 = tmp_11*pow(tmp_3, 6);
    double tmp_13 = tmp_11*tmp_4 - pow(tmp_3, 15.0/2.0)*tmp_7;
    double tmp_14 = G*m;
    double tmp_15 = tmp_14/tmp_9;
    double tmp_16 = tmp_15/pow(tmp_3, 17.0/2.0);
    double tmp_17 = x*y;
    double tmp_18 = 4*tmp_15/pow(tmp_3, 3.0/2.0);
    double tmp_19 = 3*tmp_17;
    double tmp_20 = r_s*tmp_15/pow(tmp_3, 2);
    double tmp_21 = tmp_14*log(tmp_10*tmp_6 + 1)/pow(tmp_3, 5.0/2.0);
    double tmp_22 = tmp_17*tmp_18 + tmp_19*tmp_20 - tmp_19*tmp_21;
    double tmp_23 = x*z;
    double tmp_24 = 3*tmp_20;
    double tmp_25 = 3*tmp_21;
    double tmp_26 = tmp_18*tmp_23 + tmp_23*tmp_24 - tmp_23*tmp_25;
    double tmp_27 = 3*tmp_8;
    double tmp_28 = 3*tmp_12;
    double tmp_29 = y*z;
    double tmp_30 = tmp_18*tmp_29 + tmp_24*tmp_29 - tmp_25*tmp_29;

    hess[0] = hess[0] + tmp_16*(tmp_0*tmp_4 - tmp_12*tmp_5 + tmp_13 + tmp_5*tmp_8);
    hess[1] = hess[1] + tmp_22;
    hess[2] = hess[2] + tmp_26;
    hess[3] = hess[3] + tmp_22;
    hess[4] = hess[4] + tmp_16*(tmp_1*tmp_27 - tmp_1*tmp_28 + tmp_1*tmp_4 + tmp_13);
    hess[5] = hess[5] + tmp_30;
    hess[6] = hess[6] + tmp_26;
    hess[7] = hess[7] + tmp_30;
    hess[8] = hess[8] + tmp_16*(tmp_13 + tmp_2*tmp_27 - tmp_2*tmp_28 + tmp_2*tmp_4);
}

/* ---------------------------------------------------------------------------
    Flattened NFW
*/
double flattenednfw_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (scale mass)
            - r_s (scale radius)
            - a (ignore)
            - b (ignore)
            - c (z flattening)
    */
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    const double v_h2 = -pars[0] * pars[1] / pars[2];
    const double u = norm3_flat_z(q, pars[5]) / pars[2];
    if (u == 0) {
        return v_h2;
    } else {
        return v_h2 * log(1 + u) / u;
    }
}

void flattenednfw_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - m (scale mass)
            - r_s (scale radius)
            - a (ignore)
            - b (ignore)
            - c (z flattening)
    */
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    const double v_h2 = pars[0] * pars[1] / pars[2];
    const double u = norm3_flat_z(q, pars[5]) / pars[2];

    const double fac = v_h2 / (u*u*u) / (pars[2]*pars[2]) * (log(1+u) - u/(1+u));

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2]/(pars[5]*pars[5]);
}

void flattenednfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (scale radius)
            - a (ignore)
            - b (ignore)
            - c (z flattening)
    */
    double G = pars[0];
    double m = pars[1];
    double r_s = pars[2];
    double c = pars[5];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(z, 2);
    double tmp_2 = pow(c, 2);
    double tmp_3 = pow(y, 2);
    double tmp_4 = tmp_1 + tmp_2*(tmp_0 + tmp_3);
    double tmp_5 = pow(tmp_4, 4);
    double tmp_6 = 3*tmp_0;
    double tmp_7 = tmp_4/tmp_2;
    double tmp_8 = sqrt(tmp_7);
    double tmp_9 = r_s + tmp_8;
    double tmp_10 = pow(c, 8);
    double tmp_11 = tmp_10*tmp_9;
    double tmp_12 = tmp_11*pow(tmp_7, 7.0/2.0);
    double tmp_13 = pow(tmp_4, 3);
    double tmp_14 = pow(tmp_9, 2);
    double tmp_15 = tmp_14*log(tmp_9/r_s);
    double tmp_16 = tmp_15*tmp_2;
    double tmp_17 = -tmp_11*pow(tmp_7, 9.0/2.0) + tmp_15*tmp_5;
    double tmp_18 = G*m/tmp_14;
    double tmp_19 = tmp_18/pow(tmp_7, 11.0/2.0);
    double tmp_20 = tmp_19/tmp_10;
    double tmp_21 = pow(c, 4);
    double tmp_22 = pow(tmp_4, 2);
    double tmp_23 = 3*tmp_9;
    double tmp_24 = pow(tmp_7, 3.0/2.0);
    double tmp_25 = tmp_18*x;
    double tmp_26 = tmp_21*tmp_25*y*(-3*tmp_15*tmp_21*tmp_24 + tmp_21*pow(tmp_7, 5.0/2.0) + tmp_22*tmp_23)/tmp_5;
    double tmp_27 = 3*tmp_16;
    double tmp_28 = tmp_2*z*(tmp_2*tmp_24 + tmp_23*tmp_4 - tmp_27*tmp_8)/tmp_13;
    double tmp_29 = tmp_25*tmp_28;
    double tmp_30 = tmp_18*tmp_28*y;

    hess[0] = hess[0] + tmp_20*(tmp_0*tmp_5 + tmp_12*tmp_6 - tmp_13*tmp_16*tmp_6 + tmp_17);
    hess[1] = hess[1] + tmp_26;
    hess[2] = hess[2] + tmp_29;
    hess[3] = hess[3] + tmp_26;
    hess[4] = hess[4] + tmp_20*(3*tmp_12*tmp_3 - tmp_13*tmp_27*tmp_3 + tmp_17 + tmp_3*tmp_5);
    hess[5] = hess[5] + tmp_30;
    hess[6] = hess[6] + tmp_29;
    hess[7] = hess[7] + tmp_30;
    hess[8] = hess[8] + tmp_13*tmp_19*(tmp_1*tmp_2*tmp_23*tmp_8 - tmp_1*tmp_27 + tmp_1*tmp_4 + tmp_16*tmp_4 - tmp_22*tmp_9/tmp_8)/pow(c, 12);

}

/* ---------------------------------------------------------------------------
    Triaxial NFW - triaxiality in potential!
*/
double triaxialnfw_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (scale mass)
            - r_s (scale radius)
            - a (major axis)
            - b (intermediate axis)
            - c (minor axis)
    */
    double u, v_h2;
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    v_h2 = -pars[0] * pars[1] / pars[2];
    u = sqrt(q[0]*q[0]/(pars[3]*pars[3])
           + q[1]*q[1]/(pars[4]*pars[4])
           + q[2]*q[2]/(pars[5]*pars[5])) / pars[2];

    if (u == 0) {
        return v_h2;
    } else {
        return v_h2 * log(1 + u) / u;
    }
}

void triaxialnfw_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (circular velocity at the scale radius)
            - r_s (scale radius)
            - a (major axis)
            - b (intermediate axis)
            - c (minor axis)
    */
    double fac, u, v_h2;
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    v_h2 = pars[0] * pars[1] / pars[2];
    u = sqrt(q[0]*q[0]/(pars[3]*pars[3])
           + q[1]*q[1]/(pars[4]*pars[4])
           + q[2]*q[2]/(pars[5]*pars[5])) / pars[2];

    fac = v_h2 / (u*u*u) / (pars[2]*pars[2]) * (log(1+u) - u/(1+u));

    grad[0] = grad[0] + fac*q[0]/(pars[3]*pars[3]);
    grad[1] = grad[1] + fac*q[1]/(pars[4]*pars[4]);
    grad[2] = grad[2] + fac*q[2]/(pars[5]*pars[5]);
}

void triaxialnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (scale radius)
            - a (major axis)
            - b (intermediate axis)
            - c (minor axis)
    */
    double G = pars[0];
    double m = pars[1];
    double r_s = pars[2];
    double a = pars[3];
    double b = pars[4];
    double c = pars[5];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(a, -2);
    double tmp_1 = G*m;
    double tmp_2 = tmp_0*tmp_1;
    double tmp_3 = pow(x, 2);
    double tmp_4 = pow(b, -2);
    double tmp_5 = pow(y, 2);
    double tmp_6 = pow(c, -2);
    double tmp_7 = pow(z, 2);
    double tmp_8 = tmp_0*tmp_3 + tmp_4*tmp_5 + tmp_6*tmp_7;
    double tmp_9 = pow(tmp_8, -3.0/2.0);
    double tmp_10 = 1.0/r_s;
    double tmp_11 = tmp_10*sqrt(tmp_8) + 1;
    double tmp_12 = log(tmp_11);
    double tmp_13 = tmp_12*tmp_9;
    double tmp_14 = tmp_3/pow(a, 4);
    double tmp_15 = 3*tmp_1;
    double tmp_16 = tmp_12/pow(tmp_8, 5.0/2.0);
    double tmp_17 = tmp_15*tmp_16;
    double tmp_18 = tmp_10/tmp_11;
    double tmp_19 = tmp_18/tmp_8;
    double tmp_20 = tmp_9/(pow(r_s, 2)*pow(tmp_11, 2));
    double tmp_21 = tmp_1*tmp_20;
    double tmp_22 = tmp_18/pow(tmp_8, 2);
    double tmp_23 = tmp_15*tmp_22;
    double tmp_24 = tmp_4*y;
    double tmp_25 = tmp_2*x;
    double tmp_26 = 3*tmp_25;
    double tmp_27 = tmp_16*tmp_26;
    double tmp_28 = tmp_20*tmp_25;
    double tmp_29 = tmp_22*tmp_26;
    double tmp_30 = -tmp_24*tmp_27 + tmp_24*tmp_28 + tmp_24*tmp_29;
    double tmp_31 = tmp_6*z;
    double tmp_32 = -tmp_27*tmp_31 + tmp_28*tmp_31 + tmp_29*tmp_31;
    double tmp_33 = tmp_1*tmp_13;
    double tmp_34 = tmp_5/pow(b, 4);
    double tmp_35 = tmp_1*tmp_19;
    double tmp_36 = tmp_24*tmp_31;
    double tmp_37 = -tmp_17*tmp_36 + tmp_21*tmp_36 + tmp_23*tmp_36;
    double tmp_38 = tmp_7/pow(c, 4);

    hess[0] = hess[0] + tmp_13*tmp_2 - tmp_14*tmp_17 + tmp_14*tmp_21 + tmp_14*tmp_23 - tmp_19*tmp_2;
    hess[1] = hess[1] + tmp_30;
    hess[2] = hess[2] + tmp_32;
    hess[3] = hess[3] + tmp_30;
    hess[4] = hess[4] + -tmp_17*tmp_34 + tmp_21*tmp_34 + tmp_23*tmp_34 + tmp_33*tmp_4 - tmp_35*tmp_4;
    hess[5] = hess[5] + tmp_37;
    hess[6] = hess[6] + tmp_32;
    hess[7] = hess[7] + tmp_37;
    hess[8] = hess[8] + -tmp_17*tmp_38 + tmp_21*tmp_38 + tmp_23*tmp_38 + tmp_33*tmp_6 - tmp_35*tmp_6;
}

/* ---------------------------------------------------------------------------
    Satoh potential
*/
double satoh_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    const double S2 = norm3_sq(q) + pars[2]*(pars[2] + 2*sqrt(q[2]*q[2] + pars[3]*pars[3]));
    return -pars[0] * pars[1] / sqrt(S2);
}

void satoh_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */

    const double S2 = norm3_sq(q) + pars[2]*(pars[2] + 2*sqrt(q[2]*q[2] + pars[3]*pars[3]));
    const double dPhi_dS = pars[0] * pars[1] / S2;

    grad[0] = grad[0] + dPhi_dS*q[0]/sqrt(S2);
    grad[1] = grad[1] + dPhi_dS*q[1]/sqrt(S2);
    grad[2] = grad[2] + dPhi_dS/sqrt(S2) * q[2]*(1 + pars[2] / sqrt(q[2]*q[2] + pars[3]*pars[3]));
}

double satoh_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    double z2b2 = q[2]*q[2] + pars[3]*pars[3];
    double xyz2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    double S2 = xyz2 + pars[2]*(pars[2] + 2*sqrt(z2b2));
    double A = pars[1] * pars[2] * pars[3]*pars[3] / (4*M_PI*S2*sqrt(S2)*z2b2);
    return A * (1/sqrt(z2b2) + 3/pars[2]*(1 - xyz2/S2));
}

void satoh_hessian(double t, double *pars, double *q, int n_dim, double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a ()
            - b ()
    */
    double G = pars[0];
    double m = pars[1];
    double a = pars[2];
    double b = pars[3];

    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(y, 2);
    double tmp_2 = pow(z, 2);
    double tmp_3 = pow(b, 2) + tmp_2;
    double tmp_4 = sqrt(tmp_3);
    double tmp_5 = a*(a + 2*tmp_4) + tmp_0 + tmp_1 + tmp_2;
    double tmp_6 = G*m;
    double tmp_7 = tmp_6/pow(tmp_5, 3.0/2.0);
    double tmp_8 = tmp_6/pow(tmp_5, 5.0/2.0);
    double tmp_9 = 3*tmp_8;
    double tmp_10 = -tmp_9*x*y;
    double tmp_11 = 3*z;
    double tmp_12 = a/tmp_4;
    double tmp_13 = tmp_8*(-tmp_11*tmp_12 - tmp_11);
    double tmp_14 = tmp_13*x;
    double tmp_15 = tmp_13*y;

    hess[0] = hess[0] + -tmp_0*tmp_9 + tmp_7;
    hess[1] = hess[1] + tmp_10;
    hess[2] = hess[2] + tmp_14;
    hess[3] = hess[3] + tmp_10;
    hess[4] = hess[4] + -tmp_1*tmp_9 + tmp_7;
    hess[5] = hess[5] + tmp_15;
    hess[6] = hess[6] + tmp_14;
    hess[7] = hess[7] + tmp_15;
    hess[8] = hess[8] + -tmp_13*(-tmp_12*z - z) - tmp_7*(a*tmp_2/pow(tmp_3, 3.0/2.0) - tmp_12 - 1);
}

/* ---------------------------------------------------------------------------
    Kuzmin potential
*/
double kuzmin_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
    */
    double S2 = q[0]*q[0] + q[1]*q[1] + pow(pars[2] + fabs(q[2]), 2);
    return -pars[0] * pars[1] / sqrt(S2);
}

void kuzmin_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
    */

    double S2 = q[0]*q[0] + q[1]*q[1] + pow(pars[2] + fabs(q[2]), 2);
    double fac = pars[0] * pars[1] * pow(S2, -1.5);
    double zsign;

    if (q[2] > 0) {
        zsign = 1.;
    } else if (q[2] < 0) {
        zsign = -1.;
    } else {
        zsign = 0.;
    }

    grad[0] = grad[0] + fac * q[0];
    grad[1] = grad[1] + fac * q[1];
    grad[2] = grad[2] + fac * zsign * (pars[2] + fabs(q[2]));
}

double kuzmin_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
    */
    if (q[2] != 0.) {
        return 0.;
    } else {
        return pars[1] * pars[2] / (2 * M_PI) *
            pow(q[0]*q[0] + q[1]*q[1] + pars[2]*pars[2], -1.5);
   }

}

/* ---------------------------------------------------------------------------
    Miyamoto-Nagai flattened potential
*/
double miyamotonagai_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    const double zd = (pars[2] + sqrt(q[2]*q[2] + pars[3]*pars[3]));
    return -pars[0] * pars[1] / sqrt(q[0]*q[0] + q[1]*q[1] + zd*zd);
}

void miyamotonagai_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    const double sqrtz = sqrt(q[2]*q[2] + pars[3]*pars[3]);
    const double zd = pars[2] + sqrtz;
    const double fac = pars[0]*pars[1] * pow(q[0]*q[0] + q[1]*q[1] + zd*zd, -1.5);

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2] * (1. + pars[2] / sqrtz);
}

double miyamotonagai_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */

    const double M = pars[1];
    const double a = pars[2];
    const double b = pars[3];

    const double R2 = q[0]*q[0] + q[1]*q[1];
    const double sqrt_zb = sqrt(q[2]*q[2] + b*b);
    const double numer = (b*b*M / (4*M_PI)) * (a*R2 + (a + 3*sqrt_zb)*(a + sqrt_zb)*(a + sqrt_zb));
    const double denom = pow(R2 + (a + sqrt_zb)*(a + sqrt_zb), 2.5) * sqrt_zb*sqrt_zb*sqrt_zb;

    return numer / denom;
}

void miyamotonagai_hessian(double t, double *pars, double *q, int n_dim,
                           double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    double G = pars[0];
    double m = pars[1];
    double a = pars[2];
    double b = pars[3];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(x, 2);
    double tmp_1 = pow(y, 2);
    double tmp_2 = pow(z, 2);
    double tmp_3 = pow(b, 2) + tmp_2;
    double tmp_4 = sqrt(tmp_3);
    double tmp_5 = a + tmp_4;
    double tmp_6 = pow(tmp_5, 2);
    double tmp_7 = tmp_0 + tmp_1 + tmp_6;
    double tmp_8 = G*m;
    double tmp_9 = tmp_8/pow(tmp_7, 3.0/2.0);
    double tmp_10 = 3*tmp_8/pow(tmp_7, 5.0/2.0);
    double tmp_11 = tmp_10*x;
    double tmp_12 = -tmp_11*y;
    double tmp_13 = tmp_5/tmp_4;
    double tmp_14 = tmp_13*z;
    double tmp_15 = -tmp_11*tmp_14;
    double tmp_16 = -tmp_10*tmp_14*y;
    double tmp_17 = 1.0/tmp_3;
    double tmp_18 = tmp_2*tmp_9;

    hess[0] = hess[0] + -tmp_0*tmp_10 + tmp_9;
    hess[1] = hess[1] + tmp_12;
    hess[2] = hess[2] + tmp_15;
    hess[3] = hess[3] + tmp_12;
    hess[4] = hess[4] + -tmp_1*tmp_10 + tmp_9;
    hess[5] = hess[5] + tmp_16;
    hess[6] = hess[6] + tmp_15;
    hess[7] = hess[7] + tmp_16;
    hess[8] = hess[8] + -tmp_10*tmp_17*tmp_2*tmp_6 + tmp_13*tmp_9 + tmp_17*tmp_18 - tmp_18*tmp_5/pow(tmp_3, 3.0/2.0);
}

/* ---------------------------------------------------------------------------
    MN3 exponential disk approximation

    pars:
    - G (Gravitational constant)
    - m1, a1, b1
    - m2, a2, b2
    - m3, a3, b3
*/
double mn3_value(double t, double *pars, double *q, int n_dim, void *state) {
    double tmp_pars[4] = {0., 0., 0., 0.};
    tmp_pars[0] = pars[0];

    double val = 0.;
    for (int i=0; i < 3; i++) {
        tmp_pars[1] = pars[1+3*i];
        tmp_pars[2] = pars[1+3*i+1];
        tmp_pars[3] = pars[1+3*i+2];
        val += miyamotonagai_value(t, &tmp_pars[0], q, n_dim, state);
    }
    return val;
}

void mn3_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    double tmp_pars[4] = {0., 0., 0., 0.};
    tmp_pars[0] = pars[0];

    for (int i=0; i < 3; i++) {
        tmp_pars[1] = pars[1+3*i];
        tmp_pars[2] = pars[1+3*i+1];
        tmp_pars[3] = pars[1+3*i+2];
        miyamotonagai_gradient_single(t, &tmp_pars[0], q, n_dim, grad, state);
    }
}

double mn3_density(double t, double *pars, double *q, int n_dim, void *state) {
    double tmp_pars[4] = {0., 0., 0., 0.};
    tmp_pars[0] = pars[0];

    double val = 0.;
    for (int i=0; i < 3; i++) {
        tmp_pars[1] = pars[1+3*i];
        tmp_pars[2] = pars[1+3*i+1];
        tmp_pars[3] = pars[1+3*i+2];
        val += miyamotonagai_density(t, &tmp_pars[0], q, n_dim, state);
    }
    return val;
}

void mn3_hessian(double t, double *pars, double *q, int n_dim,
                 double *hess, void *state) {
    double tmp_pars[4] = {0., 0., 0., 0.};
    tmp_pars[0] = pars[0];

    for (int i=0; i < 3; i++) {
        tmp_pars[1] = pars[1+3*i];
        tmp_pars[2] = pars[1+3*i+1];
        tmp_pars[3] = pars[1+3*i+2];
        miyamotonagai_hessian(t, &tmp_pars[0], q, n_dim, hess, state);
    }
}

/* ---------------------------------------------------------------------------
    Lee-Suto triaxial NFW from Lee & Suto (2003)
*/
double leesuto_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars: (alpha = 1)
            0 - G
            1 - v_c
            2 - r_s
            3 - a
            4 - b
            5 - c
    */
    double x, y, z, _r, u, phi0;
    double e_b2 = 1-pow(pars[4]/pars[3],2);
    double e_c2 = 1-pow(pars[5]/pars[3],2);
    double F1,F2,F3,costh2,sinth2,sinph2;

    phi0 = pars[1]*pars[1] / (log(2.) - 0.5 + (log(2.)-0.75)*e_b2 + (log(2.)-0.75)*e_c2);

    x = q[0];
    y = q[1];
    z = q[2];

    _r = sqrt(x*x + y*y + z*z);
    u = _r / pars[2];

    F1 = -log(1+u)/u;
    F2 = -1/3. + (2*u*u - 3*u + 6)/(6*u*u) + (1/u - pow(u,-3.))*log(1+u);
    F3 = (u*u - 3*u - 6)/(2*u*u*(1+u)) + 3*pow(u,-3)*log(1+u);
    costh2 = z*z / (_r*_r);
    sinth2 = 1 - costh2;
    sinph2 = y*y / (x*x + y*y);
    //return phi0 * ((e_b2/2 + e_c2/2)*((1/u - 1/(u*u*u))*log(u + 1) - 1 + (2*u*u - 3*u + 6)/(6*u*u)) + (e_b2*y*y/(2*_r*_r) + e_c2*z*z/(2*_r*_r))*((u*u - 3*u - 6)/(2*u*u*(u + 1)) + 3*log(u + 1)/(u*u*u)) - log(u + 1)/u);
    if (u == 0) {
        return phi0;
    } else {
        return phi0 * (F1 + (e_b2+e_c2)/2.*F2 + (e_b2*sinth2*sinph2 + e_c2*costh2)/2. * F3);
    }
}

void leesuto_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars: (alpha = 1)
            0 - G
            1 - v_c
            2 - r_s
            3 - a
            4 - b
            5 - c
    */
    double x, y, z, _r, _r2, _r4, ax, ay, az;
    double v_h2, x0, x2, x22;
    double x20, x21, x7, x1;
    double x10, x13, x15, x16, x17;
    double e_b2 = 1-pow(pars[4]/pars[3],2);
    double e_c2 = 1-pow(pars[5]/pars[3],2);

    v_h2 = pars[1]*pars[1] / (log(2.) - 0.5 + (log(2.)-0.75)*e_b2 + (log(2.)-0.75)*e_c2);

    x = q[0];
    y = q[1];
    z = q[2];

    _r2 = x*x + y*y + z*z;
    _r = sqrt(_r2);
    _r4 = _r2*_r2;

    x0 = _r + pars[2];
    x1 = x0*x0;
    x2 = v_h2/(12.*_r4*_r2*_r*x1);
    x10 = log(x0/pars[2]);

    x13 = _r*3.*pars[2];
    x15 = x13 - _r2;
    x16 = x15 + 6.*(pars[2]*pars[2]);
    x17 = 6.*pars[2]*x0*(_r*x16 - x0*x10*6.*(pars[2]*pars[2]));
    x20 = x0*_r2;
    x21 = 2.*_r*x0;
    x7 = e_b2*y*y + e_c2*z*z;
    x22 = -12.*_r4*_r*pars[2]*x0 + 12.*_r4*pars[2]*x1*x10 + 3.*pars[2]*x7*(x16*_r2 - 18.*x1*x10*(pars[2]*pars[2]) + x20*(2.*_r - 3.*pars[2]) + x21*(x15 + 9.*(pars[2]*pars[2]))) - x20*(e_b2 + e_c2)*(-6.*_r*pars[2]*(_r2 - (pars[2]*pars[2])) + 6.*pars[2]*x0*x10*(_r2 - 3.*(pars[2]*pars[2])) + x20*(-4.*_r + 3.*pars[2]) + x21*(-x13 + 2.*_r2 + 6.*(pars[2]*pars[2])));

    ax = x2*x*(x17*x7 + x22);
    ay = x2*y*(x17*(x7 - _r2*e_b2) + x22);
    az = x2*z*(x17*(x7 - _r2*e_c2) + x22);

    grad[0] = grad[0] + ax;
    grad[1] = grad[1] + ay;
    grad[2] = grad[2] + az;
}

double leesuto_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars: (alpha = 1)
            0 - G
            1 - v_c
            2 - r_s
            3 - a
            4 - b
            5 - c
    */
    double x, y, z, u, v_h2;
    double b_a2, c_a2;
    b_a2 = pars[4]*pars[4] / (pars[3]*pars[3]);
    c_a2 = pars[5]*pars[5] / (pars[3]*pars[3]);
    double e_b2 = 1-b_a2;
    double e_c2 = 1-c_a2;
    v_h2 = pars[1]*pars[1] / (log(2.) - 0.5 + (log(2.)-0.75)*e_b2 + (log(2.)-0.75)*e_c2);

    x = q[0];
    y = q[1];
    z = q[2];

    u = sqrt(x*x + y*y/b_a2 + z*z/c_a2) / pars[2];
    return v_h2 / (u * (1+u)*(1+u)) / (4.*M_PI*pars[2]*pars[2]*pars[0]);
}

/* ---------------------------------------------------------------------------
    Logarithmic (triaxial)
*/
double logarithmic_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (velocity scale)
            - r_h (length scale)
            - q1
            - q2
            - q3
    */
    const double x = q[0]*cos(pars[6]) + q[1]*sin(pars[6]);
    const double y = -q[0]*sin(pars[6]) + q[1]*cos(pars[6]);
    const double z = q[2];

    return 0.5*pars[1]*pars[1] * log(pars[2]*pars[2] + // scale radius
                                     x*x/(pars[3]*pars[3]) +
                                     y*y/(pars[4]*pars[4]) +
                                     z*z/(pars[5]*pars[5]));
}

double logarithmic_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (velocity scale)
            - r_h (length scale)
            - q1
            - q2
            - q3
    */
    double tmp_0 = pow(pars[3], 2);
    double tmp_1 = pow(pars[4], 2);
    double tmp_2 = tmp_0*tmp_1;
    double tmp_3 = tmp_2*pow(q[2], 2);
    double tmp_4 = pow(pars[5], 2);
    double tmp_5 = tmp_0*tmp_4;
    double tmp_6 = tmp_5*pow(q[1], 2);
    double tmp_7 = tmp_1*tmp_4;
    double tmp_8 = tmp_7*pow(q[0], 2);
    double tmp_9 = pow(pars[2], 2)*tmp_2*tmp_4;
    double tmp_10 = tmp_6 + tmp_8 + tmp_9;
    double tmp_11 = tmp_3 + tmp_9;
    return pow(pars[1], 2)*(tmp_2*(tmp_10 - tmp_3) + tmp_5*(tmp_11 - tmp_6 + tmp_8) + tmp_7*(tmp_11 + tmp_6 - tmp_8))/pow(tmp_10 + tmp_3, 2) / (4*M_PI*pars[0]);
}

void logarithmic_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (velocity scale)
            - r_h (length scale)
            - q1
            - q2
            - q3
    */

    const double x = q[0]*cos(pars[6]) + q[1]*sin(pars[6]);
    const double y = -q[0]*sin(pars[6]) + q[1]*cos(pars[6]);
    const double z = q[2];

    const double fac = pars[1]*pars[1] / (pars[2]*pars[2] + x*x/(pars[3]*pars[3]) + y*y/(pars[4]*pars[4]) + z*z/(pars[5]*pars[5]));
    const double ax = fac*x/(pars[3]*pars[3]);
    const double ay = fac*y/(pars[4]*pars[4]);
    const double az = fac*z/(pars[5]*pars[5]);

    grad[0] = grad[0] + (ax*cos(pars[6]) - ay*sin(pars[6]));
    grad[1] = grad[1] + (ax*sin(pars[6]) + ay*cos(pars[6]));
    grad[2] = grad[2] + az;
}

void logarithmic_hessian(double t, double *pars, double *q, int n_dim,
                         double *hess, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (velocity scale)
            - r_h (length scale)
            - q1
            - q2
            - q3
    */
    double v_c = pars[1];
    double r_h = pars[2];
    double q1 = pars[3];
    double q2 = pars[4];
    double q3 = pars[5];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = pow(q1, -2);
    double tmp_1 = pow(v_c, 2);
    double tmp_2 = tmp_0*tmp_1;
    double tmp_3 = pow(x, 2);
    double tmp_4 = pow(q2, -2);
    double tmp_5 = pow(y, 2);
    double tmp_6 = pow(q3, -2);
    double tmp_7 = pow(z, 2);
    double tmp_8 = pow(r_h, 2) + tmp_0*tmp_3 + tmp_4*tmp_5 + tmp_6*tmp_7;
    double tmp_9 = 1.0/tmp_8;
    double tmp_10 = 2.0/pow(tmp_8, 2);
    double tmp_11 = tmp_1*tmp_10;
    double tmp_12 = tmp_4*y;
    double tmp_13 = tmp_10*tmp_2*x;
    double tmp_14 = tmp_12*tmp_13;
    double tmp_15 = tmp_6*z;
    double tmp_16 = tmp_13*tmp_15;
    double tmp_17 = tmp_1*tmp_9;
    double tmp_18 = tmp_11*tmp_12*tmp_15;

    // minus signs because I initially borked the sympy definition
    hess[0] = hess[0] - (-tmp_2*tmp_9 + tmp_11*tmp_3/pow(q1, 4));
    hess[1] = hess[1] - (tmp_14);
    hess[2] = hess[2] - (tmp_16);
    hess[3] = hess[3] - (tmp_14);
    hess[4] = hess[4] - (-tmp_17*tmp_4 + tmp_11*tmp_5/pow(q2, 4));
    hess[5] = hess[5] - (tmp_18);
    hess[6] = hess[6] - (tmp_16);
    hess[7] = hess[7] - (tmp_18);
    hess[8] = hess[8] - (-tmp_17*tmp_6 + tmp_11*tmp_7/pow(q3, 4));
}

/* ---------------------------------------------------------------------------
    Logarithmic (triaxial)
*/
double longmuralibar_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  http://adsabs.harvard.edu/abs/1992ApJ...397...44L

        pars:
        - G (Gravitational constant)
        - m (mass scale)
        - a
        - b
        - c
        - alpha
    */
    const double x = q[0]*cos(pars[5]) + q[1]*sin(pars[5]);
    const double y = -q[0]*sin(pars[5]) + q[1]*cos(pars[5]);
    const double z = q[2];

    const double a = pars[2];
    const double b = pars[3];
    const double c = pars[4];

    const double Tm = sqrt((a-x)*(a-x) + y*y + pow(b + sqrt(c*c + z*z),2));
    const double Tp = sqrt((a+x)*(a+x) + y*y + pow(b + sqrt(c*c + z*z),2));

    return pars[0]*pars[1]/(2*a) * log((x - a + Tm) / (x + a + Tp));
}

void longmuralibar_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  http://adsabs.harvard.edu/abs/1992ApJ...397...44L

        pars:
        - G (Gravitational constant)
        - m (mass scale)
        - a
        - b
        - c
        - alpha
    */
    const double x = q[0]*cos(pars[5]) + q[1]*sin(pars[5]);
    const double y = -q[0]*sin(pars[5]) + q[1]*cos(pars[5]);
    const double z = q[2];

    const double a = pars[2];
    const double b = pars[3];
    const double c = pars[4];

    const double bcz = b + sqrt(c*c + z*z);
    const double Tm = sqrt((a-x)*(a-x) + y*y + bcz*bcz);
    const double Tp = sqrt((a+x)*(a+x) + y*y + bcz*bcz);

    const double fac1 = pars[0]*pars[1] / (2*Tm*Tp);
    const double fac2 = 1 / (y*y + bcz*bcz);
    const double fac3 = Tp + Tm - (4*x*x)/(Tp+Tm);

    const double gx = 4 * fac1 * x / (Tp + Tm);
    const double gy = fac1 * y * fac2 * fac3;
    const double gz = fac1 * z * fac2 * fac3 * bcz / sqrt(z*z + c*c);

    grad[0] = grad[0] + (gx*cos(pars[5]) - gy*sin(pars[5]));
    grad[1] = grad[1] + (gx*sin(pars[5]) + gy*cos(pars[5]));
    grad[2] = grad[2] + gz;
}

double longmuralibar_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*
        Generated by sympy...

        pars:
        - G (Gravitational constant)
        - m (mass scale)
        - a
        - b
        - c
        - alpha
    */
    double a = pars[2];
    double b = pars[3];
    double c = pars[4];

    double x = q[0]*cos(pars[5]) + q[1]*sin(pars[5]);
    double y = -q[0]*sin(pars[5]) + q[1]*cos(pars[5]);
    double z = q[2];

    double tmp0 = a - x;
    double tmp1 = pow(tmp0, 2);
    double tmp2 = pow(y, 2);
    double tmp3 = pow(z, 2);
    double tmp4 = pow(c, 2) + tmp3;
    double tmp5 = sqrt(tmp4);
    double tmp6 = b + tmp5;
    double tmp7 = pow(tmp6, 2);
    double tmp8 = tmp2 + tmp7;
    double tmp9 = tmp1 + tmp8;
    double tmp10 = sqrt(tmp9);
    double tmp11 = -a + tmp10 + x;
    double tmp12 = 1.0/tmp11;
    double tmp13 = 1.0/tmp10;
    double tmp14 = pow(tmp9, -1.5);
    double tmp15 = 1.0/tmp4;
    double tmp16 = tmp13*tmp3;
    double tmp17 = tmp6/tmp5;
    double tmp18 = pow(tmp4, -1.5);
    double tmp19 = tmp15*tmp3*tmp7;
    double tmp20 = 2*tmp2;
    double tmp21 = a + x;
    double tmp22 = pow(tmp21, 2);
    double tmp23 = tmp22 + tmp8;
    double tmp24 = sqrt(tmp23);
    double tmp25 = 1.0/tmp24;
    double tmp26 = tmp21 + tmp24;
    double tmp27 = 1.0/tmp26;
    double tmp28 = tmp25*tmp27;
    double tmp29 = tmp11*tmp28;
    double tmp30 = tmp11*tmp27/pow(tmp23, 1.5);
    double tmp31 = 1.0/tmp23;
    double tmp32 = pow(tmp26, -2);
    double tmp33 = tmp11*tmp31*tmp32;
    double tmp34 = tmp21*tmp25 + 1;
    double tmp35 = tmp27*tmp34;
    double tmp36 = tmp13*tmp15*tmp3*tmp7;
    double tmp37 = -tmp13 + tmp29;
    double tmp38 = tmp2*tmp37;
    double tmp39 = tmp0*tmp13;
    double tmp40 = tmp11*tmp27*tmp34 + tmp39 - 1;
    return pars[1]/8.*tmp12*(2*tmp11*tmp32*pow(tmp34, 2) +
        tmp12*tmp13*tmp38 + tmp12*tmp36*tmp37 + tmp12*tmp40*(-tmp39 + 1) +
        tmp13*tmp17 - tmp13*tmp20*tmp25*tmp27 + tmp13*(-tmp1/tmp9 + 1) + tmp13 -
        tmp14*tmp19 - tmp14*tmp2 + tmp15*tmp16 - tmp15*tmp28*tmp3*tmp37*tmp7 -
        tmp15*tmp29*tmp3 + 2*tmp15*tmp3*tmp33*tmp7 - tmp16*tmp18*tmp6 -
        tmp17*tmp29 + tmp18*tmp29*tmp3*tmp6 + tmp19*tmp30 + tmp2*tmp30 +
        tmp20*tmp33 - 2*tmp25*tmp27*tmp36 - tmp28*tmp38 - tmp29*(-tmp22*tmp31 +
        1) - tmp29 - tmp35*tmp40 - tmp35*(-2*tmp0*tmp13 + 2))/(M_PI*a);
}

void longmuralibar_hessian(double t, double *pars, double *q, int n_dim,
                         double *hess, void *state) {
    /* Generated by sympy...

        pars:
        - G (Gravitational constant)
        - m (mass scale)
        - a
        - b
        - c
        - alpha
    */
    double G = pars[0];
    double m = pars[1];
    double a = pars[2];
    double b = pars[3];
    double c = pars[4];
    double alpha = pars[5];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp_0 = cos(alpha);
    double tmp_1 = tmp_0*x;
    double tmp_2 = sin(alpha);
    double tmp_3 = tmp_2*y;
    double tmp_4 = tmp_1 + tmp_3;
    double tmp_5 = a + tmp_4;
    double tmp_6 = tmp_0*tmp_5;
    double tmp_7 = tmp_0*y - tmp_2*x;
    double tmp_8 = tmp_2*tmp_7;
    double tmp_9 = -tmp_8;
    double tmp_10 = tmp_6 + tmp_9;
    double tmp_11 = pow(z, 2);
    double tmp_12 = pow(c, 2) + tmp_11;
    double tmp_13 = sqrt(tmp_12);
    double tmp_14 = b + tmp_13;
    double tmp_15 = pow(tmp_14, 2);
    double tmp_16 = tmp_15 + pow(tmp_7, 2);
    double tmp_17 = tmp_16 + pow(tmp_5, 2);
    double tmp_18 = sqrt(tmp_17);
    double tmp_19 = 1.0/tmp_18;
    double tmp_20 = tmp_10*tmp_19;
    double tmp_21 = tmp_18 + tmp_5;
    double tmp_22 = 1.0/tmp_21;
    double tmp_23 = a - tmp_1 - tmp_3;
    double tmp_24 = tmp_0*tmp_23;
    double tmp_25 = -tmp_24 + tmp_9;
    double tmp_26 = tmp_16 + pow(tmp_23, 2);
    double tmp_27 = sqrt(tmp_26);
    double tmp_28 = 1.0/tmp_27;
    double tmp_29 = tmp_25*tmp_28;
    double tmp_30 = tmp_0 + tmp_29;
    double tmp_31 = -tmp_0;
    double tmp_32 = -tmp_20 + tmp_31;
    double tmp_33 = pow(tmp_21, -2);
    double tmp_34 = -a + tmp_27 + tmp_4;
    double tmp_35 = tmp_33*tmp_34;
    double tmp_36 = tmp_22*tmp_30 + tmp_32*tmp_35;
    double tmp_37 = (1.0/2.0)*G*m/a;
    double tmp_38 = tmp_37/tmp_34;
    double tmp_39 = tmp_36*tmp_38;
    double tmp_40 = tmp_21*tmp_37/pow(tmp_34, 2);
    double tmp_41 = tmp_36*tmp_40;
    double tmp_42 = tmp_32*tmp_33;
    double tmp_43 = pow(tmp_0, 2) + pow(tmp_2, 2);
    double tmp_44 = tmp_28*tmp_43;
    double tmp_45 = pow(tmp_26, -3.0/2.0);
    double tmp_46 = tmp_25*tmp_45;
    double tmp_47 = -tmp_19*tmp_43;
    double tmp_48 = pow(tmp_17, -3.0/2.0);
    double tmp_49 = tmp_10*tmp_48;
    double tmp_50 = tmp_34/pow(tmp_21, 3);
    double tmp_51 = tmp_32*tmp_50;
    double tmp_52 = tmp_21*tmp_38;
    double tmp_53 = tmp_0*tmp_7;
    double tmp_54 = tmp_2*tmp_5;
    double tmp_55 = tmp_53 + tmp_54;
    double tmp_56 = tmp_19*tmp_55;
    double tmp_57 = tmp_2 + tmp_56;
    double tmp_58 = -tmp_2;
    double tmp_59 = tmp_2*tmp_23;
    double tmp_60 = tmp_53 - tmp_59;
    double tmp_61 = tmp_28*tmp_60;
    double tmp_62 = tmp_58 - tmp_61;
    double tmp_63 = -tmp_53;
    double tmp_64 = tmp_59 + tmp_63;
    double tmp_65 = tmp_22*tmp_46;
    double tmp_66 = -tmp_56 + tmp_58;
    double tmp_67 = tmp_33*tmp_66;
    double tmp_68 = tmp_2 + tmp_61;
    double tmp_69 = -tmp_54 + tmp_63;
    double tmp_70 = tmp_35*tmp_49;
    double tmp_71 = -2*tmp_2 - 2*tmp_56;
    double tmp_72 = tmp_39*tmp_57 + tmp_41*tmp_62 + tmp_52*(tmp_30*tmp_67 + tmp_42*tmp_68 + tmp_51*tmp_71 + tmp_64*tmp_65 - tmp_69*tmp_70);
    double tmp_73 = 1.0/tmp_13;
    double tmp_74 = tmp_14*tmp_73;
    double tmp_75 = tmp_74*z;
    double tmp_76 = tmp_19*tmp_75;
    double tmp_77 = tmp_28*tmp_75;
    double tmp_78 = tmp_19*tmp_33;
    double tmp_79 = tmp_75*tmp_78;
    double tmp_80 = 2*tmp_76;
    double tmp_81 = tmp_39*tmp_76 - tmp_41*tmp_77 + tmp_52*(-tmp_30*tmp_79 + tmp_42*tmp_77 - tmp_51*tmp_80 - tmp_65*tmp_75 + tmp_70*tmp_75);
    double tmp_82 = tmp_22*tmp_68 + tmp_35*tmp_66;
    double tmp_83 = tmp_38*tmp_82;
    double tmp_84 = tmp_40*tmp_82;
    double tmp_85 = tmp_45*tmp_60;
    double tmp_86 = tmp_50*tmp_66;
    double tmp_87 = tmp_48*tmp_55;
    double tmp_88 = tmp_52*(-tmp_22*tmp_75*tmp_85 + tmp_35*tmp_75*tmp_87 + tmp_67*tmp_77 - tmp_68*tmp_79 - tmp_80*tmp_86) + tmp_76*tmp_83 - tmp_77*tmp_84;
    double tmp_89 = tmp_22*tmp_28;
    double tmp_90 = tmp_14*tmp_89;
    double tmp_91 = tmp_73*tmp_90;
    double tmp_92 = tmp_19*tmp_35;
    double tmp_93 = tmp_74*tmp_92;
    double tmp_94 = tmp_91*z - tmp_93*z;
    double tmp_95 = tmp_11/tmp_12;
    double tmp_96 = tmp_11/pow(tmp_12, 3.0/2.0);
    double tmp_97 = tmp_15*tmp_95;
    double tmp_98 = 2*tmp_97;

    hess[0] = hess[0] + tmp_39*(tmp_0 + tmp_20) + tmp_41*(-tmp_29 + tmp_31) + tmp_52*(tmp_22*(tmp_44 + tmp_46*(tmp_24 + tmp_8)) + 2*tmp_30*tmp_42 + tmp_35*(tmp_47 - tmp_49*(-tmp_6 + tmp_8)) + tmp_51*(-2*tmp_0 - 2*tmp_20));
    hess[1] = hess[1] + tmp_72;
    hess[2] = hess[2] + tmp_81;
    hess[3] = hess[3] + tmp_72;
    hess[4] = hess[4] + tmp_52*(tmp_22*(tmp_44 + tmp_64*tmp_85) + tmp_35*(tmp_47 - tmp_69*tmp_87) + 2*tmp_67*tmp_68 + tmp_71*tmp_86) + tmp_57*tmp_83 + tmp_62*tmp_84;
    hess[5] = hess[5] + tmp_88;
    hess[6] = hess[6] + tmp_81;
    hess[7] = hess[7] + tmp_88;
    hess[8] = hess[8] + tmp_38*tmp_76*tmp_94 - tmp_40*tmp_77*tmp_94 + tmp_52*(tmp_14*tmp_92*tmp_96 - tmp_22*tmp_45*tmp_97 - tmp_28*tmp_78*tmp_98 + tmp_35*tmp_48*tmp_97 + tmp_89*tmp_95 - tmp_90*tmp_96 + tmp_91 - tmp_92*tmp_95 - tmp_93 + tmp_50*tmp_98/tmp_17);
}


/* ---------------------------------------------------------------------------
    Spherical spline interpolated potentials (Density model)
*/
#if USE_GSL == 1

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

// Structure to hold cached GSL interpolation objects
typedef struct {
    gsl_spline *spline;        // Main spline for density, mass, or potential
    gsl_interp_accel *acc;     // Accelerator for main spline
    gsl_spline *rho_r_spline;  // Spline for ρ(r) * r (used in density potential calc)
    gsl_spline *rho_r2_spline; // Spline for ρ(r) * r² (used in density gradient calc)
    gsl_interp_accel *rho_r_acc;   // Accelerator for ρ(r) * r spline
    gsl_interp_accel *rho_r2_acc;  // Accelerator for ρ(r) * r² spline
    int n_knots;
    int method;
    double *r_knots;
    double *values;
} spherical_spline_state;

double spherical_spline_density_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  Spline model where the input is density as a function of radius

        pars:
            0 - G (Gravitational constant)
            1 to 1+n_knots-1 - r_knots (radial knot locations)
            n_knots to 2*n_knots-1 - density_values (density at each knot)
    */
    const double r = norm3(q);
    spherical_spline_state *spl_state = (spherical_spline_state *)state;

    // Check bounds
    if (r < spl_state->r_knots[0] || r > spl_state->r_knots[spl_state->n_knots-1]) {
        return 0.0;  // Outside interpolation range
    }

    // Calculate enclosed mass M(r) = 4π ∫[0 to r] ρ(r') r'² dr'
    const double r_min = spl_state->r_knots[0];
    const double integral_mass = gsl_spline_eval_integ(spl_state->rho_r2_spline, r_min, r, spl_state->rho_r2_acc);
    const double M_r = 4.0 * M_PI * integral_mass;

    // Calculate potential from density
    // For spherical symmetry: Φ(r) = -G M(r) / r - 4πG ∫[r to ∞] ρ(r') r' dr'
    const double r_max = spl_state->r_knots[spl_state->n_knots-1];
    const double integral_outer = gsl_spline_eval_integ(spl_state->rho_r_spline, r, r_max, spl_state->rho_r_acc);
    return -pars[0] * M_r / r - 4.0 * M_PI * pars[0] * integral_outer;
}

void spherical_spline_density_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 to 1+n_knots-1 - r_knots (radial knot locations)
            n_knots to 2*n_knots-1 - density_values (density at each knot)
    */
    const double r = norm3(q);
    if (r == 0.0) return;

    spherical_spline_state *spl_state = (spherical_spline_state *)state;

    // Check bounds
    if (r < spl_state->r_knots[0] || r > spl_state->r_knots[spl_state->n_knots-1]) {
        return;  // Outside interpolation range
    }

    // Calculate enclosed mass M(r) = 4π ∫[0 to r] ρ(r') r'² dr'
    // Use the pre-computed ρ(r) * r² spline
    const double r_min = spl_state->r_knots[0];
    const double integral = gsl_spline_eval_integ(spl_state->rho_r2_spline, r_min, r, spl_state->rho_r2_acc);
    const double M_r = 4.0 * M_PI * integral;

    // Gradient: dΦ/dr = GM(r)/r²
    double dPhi_dr = pars[0] * M_r / (r * r);

    // Convert to Cartesian gradients
    grad[0] += dPhi_dr * q[0] / r;
    grad[1] += dPhi_dr * q[1] / r;
    grad[2] += dPhi_dr * q[2] / r;
}

double spherical_spline_density_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 to 1+n_knots-1 - r_knots (radial knot locations)
            n_knots to 2*n_knots-1 - density_values (density at each knot)
    */
    const double r = norm3(q);
    spherical_spline_state *spl_state = (spherical_spline_state *)state;

    // Check bounds
    if (r < spl_state->r_knots[0] || r > spl_state->r_knots[spl_state->n_knots-1]) {
        return 0.0;  // Outside interpolation range
    }

    // Evaluate density at position
    return gsl_spline_eval(spl_state->spline, r, spl_state->acc);
}

/* ---------------------------------------------------------------------------
    Spherical spline interpolated potentials - mass
*/
double spherical_spline_mass_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 to 1+n_knots-1 - r_knots (radial knot locations)
            n_knots to 2*n_knots-1 - mass_values (mass enclosed at each knot)
    */
    const double r = norm3(q);
    spherical_spline_state *spl_state = (spherical_spline_state *)state;

    double M_r;

    // Check bounds
    if (r < spl_state->r_knots[0]) {
        // For r < r_min, use Keplerian potential with M(r_min)
        M_r = spl_state->values[0];
        return -pars[0] * M_r / r;
    }
    if (r > spl_state->r_knots[spl_state->n_knots-1]) {
        // For r > r_max, use Keplerian potential with M(r_max)
        M_r = spl_state->values[spl_state->n_knots-1];
        return -pars[0] * M_r / r;
    }

    // Calculate potential: Φ(r) = -G ∫[r to ∞] M(r')/r'² dr'
    // For finite extent with maximum radius r_max, we assume M(r') = M(r_max) for r' > r_max
    // So: Φ(r) = -G ∫[r to r_max] M(r')/r'² dr' - G M(r_max) / r_max
    const double r_max = spl_state->r_knots[spl_state->n_knots-1];
    const double M_max = spl_state->values[spl_state->n_knots-1];

    // Use numerical integration from r to r_max
    // TODO: allow number of integration points to be a parameter
    int n_integration_points = 1000;
    const double dr = (r_max - r) / n_integration_points;
    double potential = 0.0;

    for (int i = 0; i < n_integration_points; i++) {
        double r_i = r + (i + 0.5) * dr;  // Use midpoint for better accuracy
        double M_i = gsl_spline_eval(spl_state->spline, r_i, spl_state->acc);
        potential -= pars[0] * M_i * dr / (r_i * r_i);
    }

    // Add contribution from r_max to infinity (assuming constant M = M_max)
    potential -= pars[0] * M_max / r_max;

    return potential;
}

void spherical_spline_mass_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 to 1+n_knots-1 - r_knots (radial knot locations)
            n_knots to 2*n_knots-1 - mass_values (mass enclosed at each knot)
    */
    const double r = norm3(q);
    if (r == 0.0) return;

    spherical_spline_state *spl_state = (spherical_spline_state *)state;
    double M_r;

    // Check bounds
    if (r < spl_state->r_knots[0]) {
        M_r = spl_state->values[0];
    } else if (r > spl_state->r_knots[spl_state->n_knots-1]) {
        M_r = spl_state->values[spl_state->n_knots-1];
    } else {
        M_r = gsl_spline_eval(spl_state->spline, r, spl_state->acc);
    }

    // Gradient: dΦ/dr = GM(r)/r²
    const double dPhi_dr = pars[0] * M_r / (r * r * r);

    // Convert to Cartesian gradients
    grad[0] += dPhi_dr * q[0];
    grad[1] += dPhi_dr * q[1];
    grad[2] += dPhi_dr * q[2];
}

double spherical_spline_mass_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 to 1+n_knots-1 - r_knots (radial knot locations)
            n_knots to 2*n_knots-1 - mass_values (mass enclosed at each knot)
    */
    const double r = norm3(q);
    if (r == 0.0) return 0.0;

    spherical_spline_state *spl_state = (spherical_spline_state *)state;

    // Check bounds
    if (r < spl_state->r_knots[0] || r > spl_state->r_knots[spl_state->n_knots-1]) {
        return 0.0;  // Outside interpolation range
    }

    // Calculate density using: ρ(r) = (1/4πr²) dM/dr
    const double dM_dr = gsl_spline_eval_deriv(spl_state->spline, r, spl_state->acc);
    return dM_dr / (4.0 * M_PI * r * r);
}

/* ---------------------------------------------------------------------------
    Spherical spline interpolated potentials (Potential model)
*/
double spherical_spline_potential_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 to 1+n_knots-1 - r_knots (radial knot locations)
            n_knots to 2*n_knots-1 - potential_values (potential at each knot)
    */
    const double r = norm3(q);
    spherical_spline_state *spl_state = (spherical_spline_state *)state;

    // Check bounds - extrapolate beyond grid
    if (r < spl_state->r_knots[0]) {
        // Linear extrapolation to smaller radii
        const double slope = (spl_state->values[1] - spl_state->values[0]) / (spl_state->r_knots[1] - spl_state->r_knots[0]);
        return spl_state->values[0] + slope * (r - spl_state->r_knots[0]);
    }
    if (r > spl_state->r_knots[spl_state->n_knots-1]) {
        // Assume potential goes to zero at infinity - extrapolate with 1/r behavior
        return spl_state->values[spl_state->n_knots-1] * spl_state->r_knots[spl_state->n_knots-1] / r;
    }

    // Evaluate potential at position
    return gsl_spline_eval(spl_state->spline, r, spl_state->acc);
}

void spherical_spline_potential_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 to 1+n_knots-1 - r_knots (radial knot locations)
            n_knots to 2*n_knots-1 - potential_values (potential at each knot)
    */
    const double r = norm3(q);
    if (r == 0.0) return;

    spherical_spline_state *spl_state = (spherical_spline_state *)state;
    double dPhi_dr;

    // Check bounds - extrapolate beyond grid
    if (r < spl_state->r_knots[0]) {
        // Linear extrapolation to smaller radii
        // TODO: add an option to instead use the spline derivative at the first knot
        // dPhi_dr = gsl_spline_eval_deriv(spl_state->spline, spl_state->r_knots[0], spl_state->acc);
        dPhi_dr = (spl_state->values[1] - spl_state->values[0]) / (spl_state->r_knots[1] - spl_state->r_knots[0]);
    } else if (r > spl_state->r_knots[spl_state->n_knots-1]) {
        // Assume potential goes to zero at infinity - extrapolate with 1/r behavior
        // TODO: add an option to instead use the spline derivative at the final knot
        // dPhi_dr = gsl_spline_eval_deriv(spl_state->spline, spl_state->r_knots[n_knots-1], spl_state->acc);
        dPhi_dr = -spl_state->values[spl_state->n_knots-1] * spl_state->r_knots[spl_state->n_knots-1] / (r * r);
    } else {
        // Calculate gradient: dΦ/dr
        dPhi_dr = gsl_spline_eval_deriv(spl_state->spline, r, spl_state->acc);
    }

    // Convert to Cartesian gradients
    grad[0] += dPhi_dr * q[0] / r;
    grad[1] += dPhi_dr * q[1] / r;
    grad[2] += dPhi_dr * q[2] / r;
}

double spherical_spline_potential_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            0 - G (Gravitational constant)
            1 to 1+n_knots-1 - r_knots (radial knot locations)
            n_knots to 2*n_knots-1 - potential_values (potential at each knot)
    */
    const double r = norm3(q);
    if (r == 0.0) return 0.0;

    spherical_spline_state *spl_state = (spherical_spline_state *)state;

    // Check bounds
    if (r < spl_state->r_knots[0] || r > spl_state->r_knots[spl_state->n_knots-1]) {
        return 0.0;  // Outside interpolation range
    }

    // Calculate density using Poisson equation: ∇²Φ = 4πGρ
    // For spherical symmetry: ρ = (1/4πG) [d²Φ/dr² + (2/r) dΦ/dr]
    const double dPhi_dr = gsl_spline_eval_deriv(spl_state->spline, r, spl_state->acc);
    const double d2Phi_dr2 = gsl_spline_eval_deriv2(spl_state->spline, r, spl_state->acc);

    return (d2Phi_dr2 + 2.0 * dPhi_dr / r) / (4.0 * M_PI * pars[0]);
}

#endif

/* ---------------------------------------------------------------------------
    Burkert potential
    (from Mori and Burkert 2000: https://iopscience.iop.org/article/10.1086/309140/fulltext/50172.text.html)
*/
double burkert_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - rho (mass scale)
            - r0
    */
    const double r = norm3(q);
    const double x = r / pars[2];

    // pi G rho r0^2 (pi - 2(1 - r0/r)arctan(r/r0) + 2(1 - r0/r)log(1 + r/r0) - (1 - r0/r)log(1 + (r/r0)^2))
    return -M_PI * pars[0] * pars[1] * pars[2] * pars[2] * (M_PI - 2 * (1 + 1 / x) * atan(x) + 2 * (1 + 1/x) * log(1 + x) - (1 - 1/x) * log(1 + x * x) );
}


void burkert_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - rho (mass scale)
            - r0
    */
    const double r = norm3(q);
    const double x = r / pars[2];

    const double dphi_dr = -M_PI * pars[0] * pars[1] * pars[2] / (x * x) * (2 * atan(x) - 2 * log(1 + x) - log(1 + x * x));

    grad[0] = grad[0] + dphi_dr*q[0]/r;
    grad[1] = grad[1] + dphi_dr*q[1]/r;
    grad[2] = grad[2] + dphi_dr*q[2]/r;
}


double burkert_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - rho (mass scale)
            - r0
    */
    const double r = norm3(q);
    const double x = r / pars[2];
    return pars[1] / ((1 + x) * (1 + x * x));
}

#if USE_GSL == 1

// --- Helper functions for Einasto profiles ---

static inline double _s_of_r(double r, double r_m2, double alpha) {
    // s(r) = (2/α) (r/r_-2 )^α
    if (r <= 0.0) return 0.0;
    return (2.0 / alpha) * pow(r / r_m2, alpha);
}

static inline double gamma_beta(double beta, double x1, double x2) {
    return gsl_sf_gamma_inc(beta, x2) - gsl_sf_gamma_inc(beta, x1);
}

static inline double gamma_tilde_beta(
    double beta, double x1, double x2, double alpha, double r_s
) {
    const double scale = alpha * pow(r_s, alpha) / 2.0;
    return pow(scale, beta) * gamma_beta(beta, x1, x2);
}

static inline double Gamma_beta(
    double beta, double x
) {
    // Upper incomplete gamma:
    return gsl_sf_gamma(beta) - gsl_sf_gamma_inc(beta, x);
}

static inline double Gamma_tilde_beta(
    double beta, double x, double alpha, double r_s
) {
    const double scale = pow(alpha * pow(r_s, alpha) / 2.0, beta);
    return scale * Gamma_beta(beta, x);
}

static inline double _einasto_mass_enclosed(double rho_m2, double r_m2, double alpha, double r) {
    if (r <= 0.0) return 0.0;

    const double s = _s_of_r(r, r_m2, alpha);
    const double a3 = 3.0 / alpha;

    const double Mtot = (4.0 * M_PI * rho_m2 * exp(2.0/alpha) / alpha)
                        * pow(r_m2, 3.0) * pow(alpha/2.0, a3) * gsl_sf_gamma(a3);
    return Mtot * gsl_sf_gamma_inc_P(a3, s);
}

/* ---------------------------------------------------------------------------
    Einasto profile
    See: https://arxiv.org/abs/2004.10817
*/
double einasto_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - rho_-2 - scale density (density when log slope is -2)
            - r_-2 - radius at which logarithmic slope of the density profile is equal to -2
            - alpha - shape parameter
    */
    const double G = pars[0];
    const double rho_m2 = pars[1];
    const double r_m2 = pars[2];
    const double alpha = pars[3];

    const double r = norm3(q);

    const double sr  = _s_of_r(r, r_m2, alpha);
    const double a2 = 2.0 / alpha;
    const double a3 = 3.0 / alpha;

    const double pref  = (4.0 * M_PI * G * rho_m2 * exp(2.0/alpha)) / alpha;
    const double scale = alpha * pow(r_m2, alpha) / 2.0;

    if (r == 0.0) {
        const double term1 = (pow(r_m2, 3.0) / 3.0) * pow(2.0/alpha, a3);
        const double term2 = pow(scale, a2) * gsl_sf_gamma(a2);
        return - pref * (term1 + term2);
    }
    const double term1 = pow(scale, a3) * gsl_sf_gamma(a3) * gsl_sf_gamma_inc_P(a3, sr);
    const double term2 = pow(scale, a2) * gsl_sf_gamma(a2) * gsl_sf_gamma_inc_Q(a2, sr);
    return - pref * ( (term1 / r) + term2 );

}

void einasto_gradient_single(double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad, void *__restrict__ state) {
    /*  pars:
            - G (Gravitational constant)
            - rho_-2 - scale density (density when log slope is -2)
            - r_-2 - radius at which logarithmic slope of the density profile is equal to -2
            - alpha - shape parameter
    */
    const double G = pars[0];
    const double rho_m2 = pars[1];
    const double r_m2 = pars[2];
    const double alpha = pars[3];

    const double r = norm3(q);
    if (r == 0.0) {
        return;
    }

    const double Menc = _einasto_mass_enclosed(rho_m2, r_m2, alpha, r);
    const double dphi_dr = (G * Menc) / (r * r * r);

    grad[0] = grad[0] + dphi_dr * q[0];
    grad[1] = grad[1] + dphi_dr * q[1];
    grad[2] = grad[2] + dphi_dr * q[2];
}


double einasto_density(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - rho_-2 - scale density (density when log slope is -2)
            - r_-2 - radius at which logarithmic slope of the density profile is equal to -2
            - alpha - shape parameter
    */
    const double r = norm3(q);
    const double alpha = pars[3];
    return pars[1] * exp( - _s_of_r(r, pars[2], alpha) + (2.0 / alpha));
}


/* ---------------------------------------------------------------------------
    Core Einasto potential
    See: https://arxiv.org/abs/2004.10817
*/

static inline double _cEinasto_mass_enclosed(
    double rho_s, double r_s, double alpha, double r_c, double r
) {
    if (r <= 0.0) return 0.0;

    const double s0 = _s_of_r(r_c, r_s, alpha);
    const double sr = _s_of_r(r + r_c, r_s, alpha);

    const double a1 = 1.0 / alpha;
    const double a2 = 2.0 / alpha;
    const double a3 = 3.0 / alpha;

    const double scale = alpha * pow(r_s, alpha) / 2.0;

    const double seg1 = pow(scale, a1) * (gsl_sf_gamma_inc(a1, sr) - gsl_sf_gamma_inc(a1, s0));
    const double seg2 = pow(scale, a2) * (gsl_sf_gamma_inc(a2, sr) - gsl_sf_gamma_inc(a2, s0));
    const double seg3 = pow(scale, a3) * (gsl_sf_gamma_inc(a3, sr) - gsl_sf_gamma_inc(a3, s0));

    const double pref = (4.0 * M_PI * rho_s * exp(2.0/alpha)) / alpha;

    return pref * ( seg3 + (r_c*r_c)*seg1 - 2.0*r_c*seg2 );
}

double cEinasto_value(double t, double *pars, double *q, int n_dim, void *state) {
    /*  pars:
            - G (Gravitational constant)
            - rho_s - scale density
            - r_s - scale radius
            - alpha - shape parameter
            - r_c - core radius
    */
    const double G = pars[0];
    const double rho_s = pars[1];
    const double r_s = pars[2];
    const double alpha = pars[3];
    const double r_c = pars[4];

    const double r = norm3(q);

    const double a1 = 1.0 / alpha;
    const double a2 = 2.0 / alpha;
    const double a3 = 3.0 / alpha;

    const double pref  = (4.0 * M_PI * G * rho_s * exp(2.0/alpha)) / alpha;
    const double scale = alpha * pow(r_s, alpha) / 2.0;

    const double s0 = _s_of_r(r_c, r_s, alpha);

    if (r == 0.0) {
        const double dsdr0 = 2.0 * pow(r_c, alpha - 1.0) / pow(r_s, alpha);
        const double e_m_s0 = exp(-s0);

        // (1/r)*[γ̃_{3/α} + r_c^2 γ̃_{1/α} - 2 r_c γ̃_{2/α}]  -->  ds/dr|0 * e^{-s0} * sum
        const double lim_over_r =
            dsdr0 * e_m_s0 *
            ( pow(scale, a3) * pow(s0, a3 - 1.0)
            + (r_c*r_c) * pow(scale, a1) * pow(s0, a1 - 1.0)
            - 2.0 * r_c * pow(scale, a2) * pow(s0, a2 - 1.0) );

        // Upper tilded terms at r=0: Γ̃_b(s0) = scale^b * Γ(b) * Q(b, s0)
        const double up2 = pow(scale, a2) * gsl_sf_gamma(a2) * gsl_sf_gamma_inc_Q(a2, s0);
        const double up1 = pow(scale, a1) * gsl_sf_gamma(a1) * gsl_sf_gamma_inc_Q(a1, s0);

        return -pref * ( lim_over_r + (up2 - r_c * up1) );
    }

    const double sr = _s_of_r(r + r_c, r_s, alpha);

    const double seg1 = pow(scale, a1) * (gsl_sf_gamma_inc(a1, sr) - gsl_sf_gamma_inc(a1, s0));
    const double seg2 = pow(scale, a2) * (gsl_sf_gamma_inc(a2, sr) - gsl_sf_gamma_inc(a2, s0));
    const double seg3 = pow(scale, a3) * (gsl_sf_gamma_inc(a3, sr) - gsl_sf_gamma_inc(a3, s0));

    const double up2 = pow(scale, a2) * gsl_sf_gamma(a2) * gsl_sf_gamma_inc_Q(a2, sr);
    const double up1 = pow(scale, a1) * gsl_sf_gamma(a1) * gsl_sf_gamma_inc_Q(a1, sr);

    return -pref * ( ((seg3 + (r_c*r_c)*seg1 - 2.0*r_c*seg2) / r) + (up2 - r_c * up1) );
}

void cEinasto_gradient_single(
    double t, double *__restrict__ pars, double6ptr q, int n_dim, double6ptr grad,
    void *__restrict__ state
) {
    const double G = pars[0];
    const double rho_s = pars[1];
    const double r_s = pars[2];
    const double alpha = pars[3];
    const double r_c = pars[4];

    const double r = norm3(&q[0]);
    if (r == 0.0) {
        return;
    }

    const double Menc = _cEinasto_mass_enclosed(rho_s, r_s, alpha, r_c, r);
    const double dphi_dr = (G * Menc) / (r * r * r);

    grad[0] = grad[0] + dphi_dr * q[0];
    grad[1] = grad[1] + dphi_dr * q[1];
    grad[2] = grad[2] + dphi_dr * q[2];
}

double cEinasto_density(double t, double *pars, double *q, int n_dim, void *state) {
    const double rho_s = pars[1];
    const double r_s = pars[2];
    const double alpha = pars[3];
    const double r_c = pars[4];

    const double r = norm3(q);

    return rho_s * exp( - 2.0 / alpha * (pow((r + r_c) / r_s, alpha) - 1) );
}

#endif

DEFINE_VECTORIZED_GRADIENT(burkert)
DEFINE_VECTORIZED_GRADIENT(flattenednfw)
DEFINE_VECTORIZED_GRADIENT(henon_heiles)
DEFINE_VECTORIZED_GRADIENT(hernquist)
DEFINE_VECTORIZED_GRADIENT(isochrone)
DEFINE_VECTORIZED_GRADIENT(jaffe)
DEFINE_VECTORIZED_GRADIENT(kepler)
DEFINE_VECTORIZED_GRADIENT(kuzmin)
DEFINE_VECTORIZED_GRADIENT(leesuto)
DEFINE_VECTORIZED_GRADIENT(logarithmic)
DEFINE_VECTORIZED_GRADIENT(longmuralibar)
DEFINE_VECTORIZED_GRADIENT(miyamotonagai)
DEFINE_VECTORIZED_GRADIENT(mn3)
DEFINE_VECTORIZED_GRADIENT(nan)
DEFINE_VECTORIZED_GRADIENT(null)
DEFINE_VECTORIZED_GRADIENT(plummer)
DEFINE_VECTORIZED_GRADIENT(satoh)
DEFINE_VECTORIZED_GRADIENT(sphericalnfw)
DEFINE_VECTORIZED_GRADIENT(stone)
DEFINE_VECTORIZED_GRADIENT(triaxialnfw)

#if USE_GSL == 1
DEFINE_VECTORIZED_GRADIENT(cEinasto)
DEFINE_VECTORIZED_GRADIENT(einasto)
DEFINE_VECTORIZED_GRADIENT(powerlawcutoff)
DEFINE_VECTORIZED_GRADIENT(spherical_spline_density)
DEFINE_VECTORIZED_GRADIENT(spherical_spline_mass)
DEFINE_VECTORIZED_GRADIENT(spherical_spline_potential)
#endif
