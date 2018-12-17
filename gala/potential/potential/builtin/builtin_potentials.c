#include <math.h>
#include <string.h>
#include "extra_compile_macros.h"

#if USE_GSL == 1
#include <gsl/gsl_sf_gamma.h>
#endif

double nan_density(double t, double *pars, double *q, int n_dim) { return NAN; }
double nan_value(double t, double *pars, double *q, int n_dim) { return NAN; }
void nan_gradient(double t, double *pars, double *q, int n_dim, double *grad) {}
void nan_hessian(double t, double *pars, double *q, int n_dim, double *hess) {}

double null_density(double t, double *pars, double *q, int n_dim) { return 0; }
double null_value(double t, double *pars, double *q, int n_dim) { return 0; }
void null_gradient(double t, double *pars, double *q, int n_dim, double *grad){}
void null_hessian(double t, double *pars, double *q, int n_dim, double *hess) {}

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
    double GM = pars[0]*pars[1];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp0 = x*x;
    double tmp1 = y*y;
    double tmp2 = z*z;
    double tmp3 = tmp0 + tmp1 + tmp2;
    double tmp4 = GM/pow(tmp3, 1.5);
    double tmp5 = pow(tmp3, -2.5);
    double tmp6 = 3*GM*tmp5;
    double tmp7 = 3*GM*tmp5*x;
    double tmp8 = -tmp7*y;
    double tmp9 = -tmp7*z;
    double tmp10 = -tmp6*y*z;

    hess[0] = hess[0] + -tmp0*tmp6 + tmp4;
    hess[1] = hess[1] + tmp8;
    hess[2] = hess[2] + tmp9;
    hess[3] = hess[3] + tmp8;
    hess[4] = hess[4] + -tmp1*tmp6 + tmp4;
    hess[5] = hess[5] + tmp10;
    hess[6] = hess[6] + tmp9;
    hess[7] = hess[7] + tmp10;
    hess[8] = hess[8] + -tmp2*tmp6 + tmp4;
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
            - b (core scale)
    */
    double GM = pars[0]*pars[1];
    double b = pars[2];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp0 = x*x;
    double tmp1 = y*y;
    double tmp2 = z*z;
    double tmp3 = b*b + tmp0 + tmp1 + tmp2;
    double tmp4 = sqrt(tmp3);
    double tmp5 = b + tmp4;
    double tmp6 = pow(tmp5, -2);
    double tmp7 = GM*tmp6/tmp4;
    double tmp8 = 1.0/tmp3;
    double tmp9 = pow(tmp5, -3);
    double tmp10 = 2*GM*tmp8*tmp9;
    double tmp11 = pow(tmp3, -1.5);
    double tmp12 = GM*tmp11*tmp6;
    double tmp13 = 2*GM*tmp8*tmp9*x;
    double tmp14 = GM*tmp11*tmp6*x;
    double tmp15 = -tmp13*y - tmp14*y;
    double tmp16 = -tmp13*z - tmp14*z;
    double tmp17 = y*z;
    double tmp18 = -tmp10*tmp17 - tmp12*tmp17;
    hess[0] = hess[0] + -tmp0*tmp10 - tmp0*tmp12 + tmp7;
    hess[1] = hess[1] + tmp15;
    hess[2] = hess[2] + tmp16;
    hess[3] = hess[3] + tmp15;
    hess[4] = hess[4] + -tmp1*tmp10 - tmp1*tmp12 + tmp7;
    hess[5] = hess[5] + tmp18;
    hess[6] = hess[6] + tmp16;
    hess[7] = hess[7] + tmp18;
    hess[8] = hess[8] + -tmp10*tmp2 - tmp12*tmp2 + tmp7;
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
    double GM = pars[0] * pars[1];
    double c = pars[2];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp0 = x*x;
    double tmp1 = y*y;
    double tmp2 = z*z;
    double tmp3 = tmp0 + tmp1 + tmp2;
    double tmp4 = sqrt(tmp3);
    double tmp5 = c + tmp4;
    double tmp6 = pow(tmp5, -2);
    double tmp7 = GM*tmp6/tmp4;
    double tmp8 = 1.0/tmp3;
    double tmp9 = pow(tmp5, -3);
    double tmp10 = 2*GM*tmp8*tmp9;
    double tmp11 = pow(tmp3, -1.5);
    double tmp12 = GM*tmp11*tmp6;
    double tmp13 = 2*GM*tmp8*tmp9*x;
    double tmp14 = GM*tmp11*tmp6*x;
    double tmp15 = -tmp13*y - tmp14*y;
    double tmp16 = -tmp13*z - tmp14*z;
    double tmp17 = y*z;
    double tmp18 = -tmp10*tmp17 - tmp12*tmp17;

    hess[0] = hess[0] + -tmp0*tmp10 - tmp0*tmp12 + tmp7;
    hess[1] = hess[1] + tmp15;
    hess[2] = hess[2] + tmp16;
    hess[3] = hess[3] + tmp15;
    hess[4] = hess[4] + -tmp1*tmp10 - tmp1*tmp12 + tmp7;
    hess[5] = hess[5] + tmp18;
    hess[6] = hess[6] + tmp16;
    hess[7] = hess[7] + tmp18;
    hess[8] = hess[8] + -tmp10*tmp2 - tmp12*tmp2 + tmp7;
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
    double GM = pars[0] * pars[1];
    double b = pars[2];
    double x = q[0];
    double y = q[1];
    double z = q[2];

    double tmp0 = x*x;
    double tmp1 = y*y;
    double tmp2 = z*z;
    double tmp3 = b*b + tmp0 + tmp1 + tmp2;
    double tmp4 = GM/pow(tmp3, 1.5);
    double tmp5 = pow(tmp3, -2.5);
    double tmp6 = 3*GM*tmp5;
    double tmp7 = 3*GM*tmp5*x;
    double tmp8 = -tmp7*y;
    double tmp9 = -tmp7*z;
    double tmp10 = -tmp6*y*z;

    hess[0] = hess[0] + -tmp0*tmp6 + tmp4;
    hess[1] = hess[1] + tmp8;
    hess[2] = hess[2] + tmp9;
    hess[3] = hess[3] + tmp8;
    hess[4] = hess[4] + -tmp1*tmp6 + tmp4;
    hess[5] = hess[5] + tmp10;
    hess[6] = hess[6] + tmp9;
    hess[7] = hess[7] + tmp10;
    hess[8] = hess[8] + -tmp2*tmp6 + tmp4;
}

/* ---------------------------------------------------------------------------
    Jaffe sphere

    TODO: I think this is wrong?
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
    rho0 = pars[1]/(2*M_PI*pars[2]*pars[2]*pars[2]);
    return rho0 / (pow(r/pars[2],2) * pow(1+r/pars[2],2));
}

/* ---------------------------------------------------------------------------
    Power-law potential with exponential cutoff
*/
#if USE_GSL == 1
double powerlawcutoff_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            0 - G (Gravitational constant)
            1 - m (total mass)
            2 - a (power-law index)
            3 - c (cutoff radius)
    */
    double r;
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);

    if (r == 0.) {
        return -INFINITY;
    } else {
        return -(pars[0] * pars[1] / (pars[3] * r) / gsl_sf_gamma((3-pars[2])/2) *
            (pars[3] * gsl_sf_gamma((3.-pars[2]) / 2) * gsl_sf_gamma_inc_P((3.-pars[2]) / 2, r*r/(pars[3]*pars[3])) -
                   r * gsl_sf_gamma((2.-pars[2]) / 2) * gsl_sf_gamma_inc_P((2.-pars[2]) / 2, r*r/(pars[3]*pars[3]))));
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
    A = pars[1] / (2*M_PI) * pow(pars[3], pars[2]-3) / gsl_sf_gamma(0.5 * (3-pars[2]));
    return A * pow(r, -pars[2]) * exp(-r*r / (pars[3]*pars[3]));
}

void powerlawcutoff_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            0 - G (Gravitational constant)
            1 - m (total mass)
            2 - a (power-law index)
            3 - c (cutoff radius)
    */
    double r, A, dPhi_dr;
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    dPhi_dr = (pars[0] * pars[1] / (r*r) *
        gsl_sf_gamma_inc_P(0.5 * (3-pars[2]), r*r/(pars[3]*pars[3]))); // / gsl_sf_gamma(0.5 * (3-pars[2])));

    grad[0] = grad[0] + dPhi_dr * q[0]/r;
    grad[1] = grad[1] + dPhi_dr * q[1]/r;
    grad[2] = grad[2] + dPhi_dr * q[2]/r;
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

    return -fac * (atan(u_h)/u_h - atan(u_c)/u_c +
                   0.5*log((r*r + pars[3]*pars[3])/(r*r + pars[2]*pars[2])));
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
    return v_h2 * log(1 + u) / u;
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

double sphericalnfw_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
  /*  pars:
        - G (Gravitational constant)
        - m (mass scale)
        - r_s (scale radius)
  */
  // double v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
  double v_h2 = -pars[0] * pars[1] / pars[2];
  double rs = pars[2];

  double x = q[0];
  double y = q[1];
  double z = q[2];
  double r = sqrt(x*x + y*y + z*z);
  double r2 = r*r;
  double r3 = r2*r;
  double r4 = r3*r;
  double r5 = r4*r;
  double rrs1 = r/rs + 1.0;
  double ll = log(rrs1);

  hess[0] = hess[0] + v_h2*(-1./(r2*rrs1) + rs/r3*ll + x*x/(r3*rs*rrs1*rrs1) + 3.*x*x/(r4*rrs1) - 3.*rs/r5*x*x*ll);
  hess[1] = hess[1] + v_h2*(x*y/(r3*rs*rrs1*rrs1) + 3.*x*y/(r4*rrs1) - 3.*rs/r5*x*y*ll);
  hess[2] = hess[2] + v_h2*(x*z/(r3*rs*rrs1*rrs1) + 3.*x*z/(r4*rrs1) - 3.*rs/r5*x*z*ll);
  hess[3] = hess[3] + v_h2*(x*y/(r3*rs*rrs1*rrs1) + 3.*x*y/(r4*rrs1) - 3.*rs/r5*x*y*ll);
  hess[4] = hess[4] + v_h2*(-1./(r2*rrs1) + rs/r3*ll + y*y/(r3*rs*rrs1*rrs1) + 3.*y*y/(r4*rrs1) - 3.*rs/r5*y*y*ll);
  hess[5] = hess[5] + v_h2*(y*z/(r3*rs*rrs1*rrs1) + 3.*y*z/(r4*rrs1) - 3.*rs/r5*y*z*ll);
  hess[6] = hess[6] + v_h2*(x*z/(r3*rs*rrs1*rrs1) + 3.*x*z/(r4*rrs1) - 3.*rs/r5*x*z*ll);
  hess[7] = hess[7] + v_h2*(y*z/(r3*rs*rrs1*rrs1) + 3.*y*z/(r4*rrs1) - 3.*rs/r5*y*z*ll);
  hess[8] = hess[8] + v_h2*(-1./(r2*rrs1) + rs/r3*ll + z*z/(r3*rs*rrs1*rrs1) + 3.*z*z/(r4*rrs1) - 3.*rs/r5*z*z*ll);

}

/* ---------------------------------------------------------------------------
    Flattened NFW
*/
double flattenednfw_value(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            - G (Gravitational constant)
            - m (scale mass)
            - r_s (scale radius)
            - qz (flattening)
    */
    double u, v_h2;
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    v_h2 = -pars[0] * pars[1] / pars[2];
    u = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]/(pars[3]*pars[3])) / pars[2];
    return v_h2 * log(1 + u) / u;
}

void flattenednfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (scale mass)
            - r_s (scale radius)
            - qz (flattening)
    */
    double fac, u, v_h2;
    // v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    v_h2 = pars[0] * pars[1] / pars[2];
    u = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]/(pars[3]*pars[3])) / pars[2];

    fac = v_h2 / (u*u*u) / (pars[2]*pars[2]) * (log(1+u) - u/(1+u));

    grad[0] = grad[0] + fac*q[0];
    grad[1] = grad[1] + fac*q[1];
    grad[2] = grad[2] + fac*q[2]/(pars[3]*pars[3]);
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
    return v_h2 * log(1 + u) / u;
}

void triaxialnfw_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (circular velocity at the scale radius)
            - r_s (scale radius)
            - q (flattening)
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

void miyamotonagai_hessian(double t, double *pars, double *q, int n_dim, double *hess) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    double G, M, a, b;
    G = pars[0];
    M = pars[1];
    a = pars[2];
    b = pars[3];

    double tmp0 = q[0]*q[0];
    double tmp1 = q[1]*q[1];
    double tmp2 = q[2]*q[2];
    double tmp3 = b*b + tmp2;
    double tmp4 = sqrt(tmp3);
    double tmp5 = a + tmp4;
    double tmp6 = tmp5*tmp5;
    double tmp7 = tmp0 + tmp1 + tmp6;
    double tmp8 = pow(tmp7, -1.5);
    double tmp9 = G*M*tmp8;
    double tmp10 = pow(tmp7, -2.5);
    double tmp11 = 3*G*M*tmp10;
    double tmp12 = 3*G*M*tmp10*q[0];
    double tmp13 = -tmp12*q[1];
    double tmp14 = 1.0/tmp4;
    double tmp15 = tmp14*tmp5*q[2];
    double tmp16 = -tmp12*tmp15;
    double tmp17 = -3*G*M*tmp10*tmp15*q[1];
    double tmp18 = 1.0/tmp3;
    double tmp19 = G*M*tmp2*tmp8;

    hess[0] = hess[0] + -tmp0*tmp11 + tmp9;
    hess[1] = hess[1] + tmp13;
    hess[2] = hess[2] + tmp16;

    hess[3] = hess[3] + tmp13;
    hess[4] = hess[4] + -tmp1*tmp11 + tmp9;
    hess[5] = hess[5] + tmp17;

    hess[6] = hess[6] + tmp16;
    hess[7] = hess[7] + tmp17;
    hess[8] = hess[8] + -tmp11*tmp18*tmp2*tmp6 + tmp14*tmp5*tmp9 + tmp18*tmp19 - tmp19*tmp5*pow(tmp3,-1.5);
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
    return phi0 * (F1 + (e_b2+e_c2)/2.*F2 + (e_b2*sinth2*sinph2 + e_c2*costh2)/2. * F3);
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
    /* pars[0] is G -- unused here */
    double x, y, z;

    x = q[0]*cos(pars[6]) + q[1]*sin(pars[6]);
    y = -q[0]*sin(pars[6]) + q[1]*cos(pars[6]);
    z = q[2];

    return 0.5*pars[1]*pars[1] * log(pars[2]*pars[2] + // scale radius
                                     x*x/(pars[3]*pars[3]) +
                                     y*y/(pars[4]*pars[4]) +
                                     z*z/(pars[5]*pars[5]));
}

void logarithmic_gradient(double t, double *pars, double *q, int n_dim, double *grad) {
    /* pars[0] is G -- unused here */
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

    Tm = sqrt((a-x)*(a-x) + y*y + pow(b + sqrt(c*c+z*z),2));
    Tp = sqrt((a+x)*(a+x) + y*y + pow(b + sqrt(c*c+z*z),2));

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
