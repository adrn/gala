#include <math.h>
#include <string.h>
#include <stdio.h>
#include "extra_compile_macros.h"

#if USE_GSL == 1
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_math.h>
#endif

double nan_density(double t, double *pars, double *q, int n_dim) { return NAN; }
double nan_value(double t, double *pars, double *q, int n_dim) { return NAN; }
void nan_gradient(double t, double *pars, double *q, int n_dim, double *grad) {}
void nan_hessian(double t, double *pars, double *q, int n_dim, double *hess) {}

double null_density(double t, double *pars, double *q, int n_dim) { return 0; }
double null_value(double t, double *pars, double *q, int n_dim) { return 0; }
void null_gradient(double t, double *pars, double *q, int n_dim, double *grad){}
void null_hessian(double t, double *pars, double *q, int n_dim, double *hess) {}

/* Note: many Hessians generated with sympy in
    gala-notebooks/Make-all-Hessians.ipynb
*/

/* ---------------------------------------------------------------------------
    Henon-Heiles potential
*/
double henon_heiles_value(double t, double *pars, double *q, int n_dim) {
    /*  no parameters... */
    return 0.5 * (q[0]*q[0] + q[1]*q[1] + 2*q[0]*q[0]*q[1] - 2/3.*q[1]*q[1]*q[1]);
}

void henon_heiles_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  no parameters... */
    grad[0] = grad[0] + q[0] + 2*q[0]*q[1];
    grad[1] = grad[1] + q[1] + q[0]*q[0] - q[1]*q[1];
}

void henon_heiles_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double kepler_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    double R;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    return -pars[0] * pars[1] / R;
}

void kepler_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    double R, fac;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    fac = pars[0] * pars[1] / (R*R*R);

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double kepler_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    double r2;
    r2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];

    if (r2 == 0.) {
        return INFINITY;
    } else {
        return 0.;
    }
}

void kepler_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double isochrone_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    double R2;
    R2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    return -pars[0] * pars[1] / (sqrt(R2 + pars[2]*pars[2]) + pars[2]);
}

void isochrone_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    double sqrt_r2_b2, fac, denom;
    sqrt_r2_b2 = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + pars[2]*pars[2]);
    denom = sqrt_r2_b2 * (sqrt_r2_b2 + pars[2])*(sqrt_r2_b2 + pars[2]);
    fac = pars[0] * pars[1] / denom;

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double isochrone_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    double r2, a, b;
    b = pars[2];
    r2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    a = sqrt(b*b + r2);

    return pars[1] * (3*(b+a)*a*a - r2*(b+3*a)) / (4*M_PI*pow(b+a,3)*a*a*a);
}

void isochrone_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double hernquist_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    return -pars[0] * pars[1] / (R + pars[2]);
}

void hernquist_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R, fac;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    fac = pars[0] * pars[1] / ((R + pars[2]) * (R + pars[2]) * R);

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double hernquist_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double r, rho0;
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    rho0 = pars[1]/(2*M_PI*pars[2]*pars[2]*pars[2]);
    return rho0 / ((r/pars[2]) * pow(1+r/pars[2],3));
}

void hernquist_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double plummer_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double R2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    return -pars[0]*pars[1] / sqrt(R2 + pars[2]*pars[2]);
}

void plummer_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double R2b, fac;
    R2b = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + pars[2]*pars[2];
    fac = pars[0] * pars[1] / sqrt(R2b) / R2b;

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double plummer_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double r2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    return 3*pars[1] / (4*M_PI*pars[2]*pars[2]*pars[2]) * pow(1 + r2/(pars[2]*pars[2]), -2.5);
}

void plummer_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double jaffe_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    return -pars[0] * pars[1] / pars[2] * log(1 + pars[2]/R);
}

void jaffe_gradient(double t, double *pars, double *q, int n_dim, double *grad){
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R, fac;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    fac = pars[0] * pars[1] / pars[2] * (pars[2] / (R * (pars[2] + R)));

    grad[0] = grad[0] + fac*q[0]/R;
    grad[1] = grad[1] + fac*q[1]/R;
    grad[2] = grad[2] + fac*q[2]/R;
}

double jaffe_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double r, rho0;
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    rho0 = pars[1] / (4*M_PI*pars[2]*pars[2]*pars[2]);
    return rho0 / (pow(r/pars[2],2) * pow(1+r/pars[2],2));
}

void jaffe_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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

double powerlawcutoff_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            0 - G (Gravitational constant)
            1 - m (total mass)
            2 - a (power-law index)
            3 - c (cutoff radius)
    */
    double G = pars[0];
    double m = pars[1];
    double alpha = pars[2];
    double r_c = pars[3];
    double x = q[0];
    double y = q[1];
    double z = q[2];
    double r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);

    if (r == 0.) {
        return -INFINITY;
    } else {
        double tmp_0 = (1.0/2.0)*alpha;
        double tmp_1 = -tmp_0;
        double tmp_2 = tmp_1 + 1.5;
        double tmp_3 = pow(x, 2) + pow(y, 2) + pow(z, 2);
        double tmp_4 = tmp_3/pow(r_c, 2);
        double tmp_5 = G*m;
        double tmp_6 = tmp_5*safe_gamma_inc(tmp_2, tmp_4)/(sqrt(tmp_3)*tgamma(tmp_1 + 2.5));
        return tmp_0*tmp_6 - 3.0/2.0*tmp_6 + tmp_5*safe_gamma_inc(tmp_1 + 1, tmp_4)/(r_c*tgamma(tmp_2));
    }
}

double powerlawcutoff_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            0 - G (Gravitational constant)
            1 - m (total mass)
            2 - a (power-law index)
            3 - c (cutoff radius)
    */
    double r, A;
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    A = pars[1] / (2*M_PI) * pow(pars[3], pars[2] - 3) / gsl_sf_gamma(0.5 * (3 - pars[2]));
    return A * pow(r, -pars[2]) * exp(-r*r / (pars[3]*pars[3]));
}

void powerlawcutoff_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            0 - G (Gravitational constant)
            1 - m (total mass)
            2 - a (power-law index)
            3 - c (cutoff radius)
    */
    double r, dPhi_dr;
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    dPhi_dr = (pars[0] * pars[1] / (r*r) *
        gsl_sf_gamma_inc_P(0.5 * (3-pars[2]), r*r/(pars[3]*pars[3]))); // / gsl_sf_gamma(0.5 * (3-pars[2])));

    grad[0] = grad[0] + dPhi_dr * q[0]/r;
    grad[1] = grad[1] + dPhi_dr * q[1]/r;
    grad[2] = grad[2] + dPhi_dr * q[2]/r;
}

void powerlawcutoff_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double stone_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - M (total mass)
            - r_c (core radius)
            - r_h (halo radius)
    */
    double r, u_c, u_h, fac;

    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    u_c = r / pars[2];
    u_h = r / pars[3];

    fac = 2*pars[0]*pars[1] / M_PI / (pars[3] - pars[2]);

    if (r == 0) {
        return -fac * 0.5 * log(pars[3]*pars[3] / (pars[2] * pars[2]));
    } else {
        return -fac * (
            atan(u_h)/u_h - atan(u_c)/u_c +
            0.5*log((r*r + pars[3]*pars[3])/(r*r + pars[2]*pars[2]))
        );
    }

}

void stone_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - M (total mass)
            - r_c (core radius)
            - r_h (halo radius)
    */
    double r, u_c, u_h, fac, dphi_dr;

    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    u_c = r / pars[2];
    u_h = r / pars[3];

    fac = 2*pars[0]*pars[1] / (M_PI*r*r) / (pars[2] - pars[3]);  // order flipped from value
    dphi_dr = fac * (pars[2]*atan(u_c) - pars[3]*atan(u_h));

    grad[0] = grad[0] + dphi_dr*q[0]/r;
    grad[1] = grad[1] + dphi_dr*q[1]/r;
    grad[2] = grad[2] + dphi_dr*q[2]/r;
}

double stone_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - M (total mass)
            - r_c (core radius)
            - r_h (halo radius)
    */
    double r, u_c, u_t, rho;
    rho = pars[1] * (pars[2] + pars[3]) / (2*M_PI*M_PI*pars[2]*pars[2]*pars[3]*pars[3]);

    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    u_c = r / pars[2];
    u_t = r / pars[3];

    return rho / ((1 + u_c*u_c)*(1 + u_t*u_t));
}

void stone_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double sphericalnfw_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (scale radius)
    */
    double u, v_h2;
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    v_h2 = -pars[0] * pars[1] / pars[2];
    u = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]) / pars[2];
    if (u == 0) {
        return v_h2;
    } else {
        return v_h2 * log(1 + u) / u;
    }
}

void sphericalnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (scale radius)
    */
    double fac, u, v_h2;
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    v_h2 = pars[0] * pars[1] / pars[2];

    u = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]) / pars[2];
    fac = v_h2 / (u*u*u) / (pars[2]*pars[2]) * (log(1+u) - u/(1+u));

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2];
}

double sphericalnfw_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - r_s (scale radius)
    */
    // double v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    double v_h2 = pars[0] * pars[1] / pars[2];
    double r, rho0;
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);

    rho0 = v_h2 / (4*M_PI*pars[0]*pars[2]*pars[2]);
    return rho0 / ((r/pars[2]) * pow(1+r/pars[2],2));
}

void sphericalnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double flattenednfw_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (scale mass)
            - r_s (scale radius)
            - a (ignore)
            - b (ignore)
            - c (z flattening)
    */
    double u, v_h2;
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    v_h2 = -pars[0] * pars[1] / pars[2];
    u = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]/(pars[5]*pars[5])) / pars[2];
    if (u == 0) {
        return v_h2;
    } else {
        return v_h2 * log(1 + u) / u;
    }
}

void flattenednfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (scale mass)
            - r_s (scale radius)
            - a (ignore)
            - b (ignore)
            - c (z flattening)
    */
    double fac, u, v_h2;
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    v_h2 = pars[0] * pars[1] / pars[2];
    u = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]/(pars[5]*pars[5])) / pars[2];

    fac = v_h2 / (u*u*u) / (pars[2]*pars[2]) * (log(1+u) - u/(1+u));

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2]/(pars[5]*pars[5]);
}

void flattenednfw_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double triaxialnfw_value(double t, double *pars, double *q, int n_dim) {
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

void triaxialnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
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

void triaxialnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double satoh_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    double S2;
    S2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + pars[2]*(pars[2] + 2*sqrt(q[2]*q[2] + pars[3]*pars[3]));
    return -pars[0] * pars[1] / sqrt(S2);
}

void satoh_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */

    double S2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + pars[2]*(pars[2] + 2*sqrt(q[2]*q[2] + pars[3]*pars[3]));
    double dPhi_dS = pars[0] * pars[1] / S2;

    grad[0] = grad[0] + dPhi_dS*q[0]/sqrt(S2);
    grad[1] = grad[1] + dPhi_dS*q[1]/sqrt(S2);
    grad[2] = grad[2] + dPhi_dS/sqrt(S2) * q[2]*(1 + pars[2] / sqrt(q[2]*q[2] + pars[3]*pars[3]));
}

double satoh_density(double t, double *pars, double *q, int n_dim) {
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

void satoh_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
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
double kuzmin_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
    */
    double S2 = q[0]*q[0] + q[1]*q[1] + pow(pars[2] + fabs(q[2]), 2);
    return -pars[0] * pars[1] / sqrt(S2);
}

void kuzmin_gradient(double t, double *pars, double *q, int n_dim,
                     double *grad) {
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

double kuzmin_density(double t, double *pars, double *q, int n_dim) {
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
double miyamotonagai_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    double zd;
    zd = (pars[2] + sqrt(q[2]*q[2] + pars[3]*pars[3]));
    return -pars[0] * pars[1] / sqrt(q[0]*q[0] + q[1]*q[1] + zd*zd);
}

void miyamotonagai_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    double sqrtz, zd, fac;

    sqrtz = sqrt(q[2]*q[2] + pars[3]*pars[3]);
    zd = pars[2] + sqrtz;
    fac = pars[0]*pars[1] * pow(q[0]*q[0] + q[1]*q[1] + zd*zd, -1.5);

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2] * (1. + pars[2] / sqrtz);
}

double miyamotonagai_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */

    double M, a, b;
    M = pars[1];
    a = pars[2];
    b = pars[3];

    double R2 = q[0]*q[0] + q[1]*q[1];
    double sqrt_zb = sqrt(q[2]*q[2] + b*b);
    double numer = (b*b*M / (4*M_PI)) * (a*R2 + (a + 3*sqrt_zb)*(a + sqrt_zb)*(a + sqrt_zb));
    double denom = pow(R2 + (a + sqrt_zb)*(a + sqrt_zb), 2.5) * sqrt_zb*sqrt_zb*sqrt_zb;

    return numer/denom;
}

void miyamotonagai_hessian(double t, double *pars, double *q, int n_dim,
                           double *hess) {
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
double mn3_value(double t, double *pars, double *q, int n_dim) {
    double tmp_pars[4] = {0., 0., 0., 0.};
    tmp_pars[0] = pars[0];

    double val = 0.;
    for (int i=0; i < 3; i++) {
        tmp_pars[1] = pars[1+3*i];
        tmp_pars[2] = pars[1+3*i+1];
        tmp_pars[3] = pars[1+3*i+2];
        val += miyamotonagai_value(t, &tmp_pars[0], q, n_dim);
    }
    return val;
}

void mn3_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    double tmp_pars[4] = {0., 0., 0., 0.};
    tmp_pars[0] = pars[0];

    for (int i=0; i < 3; i++) {
        tmp_pars[1] = pars[1+3*i];
        tmp_pars[2] = pars[1+3*i+1];
        tmp_pars[3] = pars[1+3*i+2];
        miyamotonagai_gradient(t, &tmp_pars[0], q, n_dim, grad);
    }
}

double mn3_density(double t, double *pars, double *q, int n_dim) {
    double tmp_pars[4] = {0., 0., 0., 0.};
    tmp_pars[0] = pars[0];

    double val = 0.;
    for (int i=0; i < 3; i++) {
        tmp_pars[1] = pars[1+3*i];
        tmp_pars[2] = pars[1+3*i+1];
        tmp_pars[3] = pars[1+3*i+2];
        val += miyamotonagai_density(t, &tmp_pars[0], q, n_dim);
    }
    return val;
}

void mn3_hessian(double t, double *pars, double *q, int n_dim,
                 double *hess) {
    double tmp_pars[4] = {0., 0., 0., 0.};
    tmp_pars[0] = pars[0];

    for (int i=0; i < 3; i++) {
        tmp_pars[1] = pars[1+3*i];
        tmp_pars[2] = pars[1+3*i+1];
        tmp_pars[3] = pars[1+3*i+2];
        miyamotonagai_hessian(t, &tmp_pars[0], q, n_dim, hess);
    }
}

/* ---------------------------------------------------------------------------
    Lee-Suto triaxial NFW from Lee & Suto (2003)
*/
double leesuto_value(double t, double *pars, double *q, int n_dim) {
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

void leesuto_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
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

double leesuto_density(double t, double *pars, double *q, int n_dim) {
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
double logarithmic_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (velocity scale)
            - r_h (length scale)
            - q1
            - q2
            - q3
    */
    double x, y, z;

    x = q[0]*cos(pars[6]) + q[1]*sin(pars[6]);
    y = -q[0]*sin(pars[6]) + q[1]*cos(pars[6]);
    z = q[2];

    return 0.5*pars[1]*pars[1] * log(pars[2]*pars[2] + // scale radius
                                     x*x/(pars[3]*pars[3]) +
                                     y*y/(pars[4]*pars[4]) +
                                     z*z/(pars[5]*pars[5]));
}

double logarithmic_density(double t, double *pars, double *q, int n_dim) {
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

void logarithmic_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (velocity scale)
            - r_h (length scale)
            - q1
            - q2
            - q3
    */
    double x, y, z, ax, ay, az, fac;

    x = q[0]*cos(pars[6]) + q[1]*sin(pars[6]);
    y = -q[0]*sin(pars[6]) + q[1]*cos(pars[6]);
    z = q[2];

    fac = pars[1]*pars[1] / (pars[2]*pars[2] + x*x/(pars[3]*pars[3]) + y*y/(pars[4]*pars[4]) + z*z/(pars[5]*pars[5]));
    ax = fac*x/(pars[3]*pars[3]);
    ay = fac*y/(pars[4]*pars[4]);
    az = fac*z/(pars[5]*pars[5]);

    grad[0] = grad[0] + (ax*cos(pars[6]) - ay*sin(pars[6]));
    grad[1] = grad[1] + (ax*sin(pars[6]) + ay*cos(pars[6]));
    grad[2] = grad[2] + az;
}

void logarithmic_hessian(double t, double *pars, double *q, int n_dim,
                         double *hess) {
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
double longmuralibar_value(double t, double *pars, double *q, int n_dim) {
    /*  http://adsabs.harvard.edu/abs/1992ApJ...397...44L

        pars:
        - G (Gravitational constant)
        - m (mass scale)
        - a
        - b
        - c
        - alpha
    */
    double x, y, z;
    double a, b, c;
    double Tm, Tp;

    x = q[0]*cos(pars[5]) + q[1]*sin(pars[5]);
    y = -q[0]*sin(pars[5]) + q[1]*cos(pars[5]);
    z = q[2];

    a = pars[2];
    b = pars[3];
    c = pars[4];

    Tm = sqrt((a-x)*(a-x) + y*y + pow(b + sqrt(c*c + z*z),2));
    Tp = sqrt((a+x)*(a+x) + y*y + pow(b + sqrt(c*c + z*z),2));

    return pars[0]*pars[1]/(2*a) * log((x - a + Tm) / (x + a + Tp));
}

void longmuralibar_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  http://adsabs.harvard.edu/abs/1992ApJ...397...44L

        pars:
        - G (Gravitational constant)
        - m (mass scale)
        - a
        - b
        - c
        - alpha
    */
    double x, y, z;
    double a, b, c;
    double Tm, Tp, fac1, fac2, fac3, bcz;
    double gx, gy, gz;

    x = q[0]*cos(pars[5]) + q[1]*sin(pars[5]);
    y = -q[0]*sin(pars[5]) + q[1]*cos(pars[5]);
    z = q[2];

    a = pars[2];
    b = pars[3];
    c = pars[4];

    bcz = b + sqrt(c*c + z*z);
    Tm = sqrt((a-x)*(a-x) + y*y + bcz*bcz);
    Tp = sqrt((a+x)*(a+x) + y*y + bcz*bcz);

    fac1 = pars[0]*pars[1] / (2*Tm*Tp);
    fac2 = 1 / (y*y + bcz*bcz);
    fac3 = Tp + Tm - (4*x*x)/(Tp+Tm);

    gx = 4 * fac1 * x / (Tp + Tm);
    gy = fac1 * y * fac2 * fac3;
    gz = fac1 * z * fac2 * fac3 * bcz / sqrt(z*z + c*c);

    grad[0] = grad[0] + (gx*cos(pars[5]) - gy*sin(pars[5]));
    grad[1] = grad[1] + (gx*sin(pars[5]) + gy*cos(pars[5]));
    grad[2] = grad[2] + gz;
}

double longmuralibar_density(double t, double *pars, double *q, int n_dim) {
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
                         double *hess) {
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
    Burkert potential
    (from Mori and Burkert 2000: https://iopscience.iop.org/article/10.1086/309140/fulltext/50172.text.html)
*/
double burkert_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - rho (mass scale)
            - r0
    */
    double R, x;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    x = R / pars[2];
    
    // pi G rho r0^2 (pi - 2(1 - r0/r)arctan(r/r0) + 2(1 - r0/r)log(1 + r/r0) - (1 - r0/r)log(1 + (r/r0)^2))
    return -M_PI * pars[0] * pars[1] * pars[2] * pars[2] * (M_PI - 2 * (1 + 1 / x) * atan(x) + 2 * (1 + 1/x) * log(1 + x) - (1 - 1/x) * log(1 + x * x) );
}


void burkert_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - rho (mass scale)
            - r0
    */
    double R, x, dphi_dr;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    x = R / pars[2];

    dphi_dr = -M_PI * pars[0] * pars[1] * pars[2] / (x * x) * (2 * atan(x) - 2 * log(1 + x) - log(1 + x * x));

    grad[0] = grad[0] + dphi_dr*q[0]/R;
    grad[1] = grad[1] + dphi_dr*q[1]/R;
    grad[2] = grad[2] + dphi_dr*q[2]/R;
}


double burkert_density(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - rho (mass scale)
            - r0
    */
    double r, x, rho;

    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    x = r / pars[2];
    rho = pars[1] / ((1 + x) * (1 + x * x));

    return rho;
}