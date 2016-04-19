#include <math.h>
#include <string.h>

double nan_density(double t, double *pars, double *q) {
    return NAN;
}

/* ---------------------------------------------------------------------------
    Henon-Heiles potential
*/
double henon_heiles_value(double t, double *pars, double *q) {
    /*  no parameters... */
    return 0.5 * (q[0]*q[0] + q[1]*q[1] + 2*q[0]*q[0]*q[1] - 2/3.*q[1]*q[1]*q[1]);
}

void henon_heiles_gradient(double t, double *pars, double *q, double *grad) {
    /*  no parameters... */
    grad[0] = grad[0] + q[0] + 2*q[0]*q[1];
    grad[1] = grad[1] + q[1] + q[0]*q[0] - q[1]*q[1];
}

/* ---------------------------------------------------------------------------
    Kepler potential
*/
double kepler_value(double t, double *pars, double *q) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    double R;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    return -pars[0] * pars[1] / R;
}

void kepler_gradient(double t, double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    double R, fac;
    R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    fac = pars[0] * pars[1] / (R*R*R);

    grad[0] = grad[0] + fac*r[0];
    grad[1] = grad[1] + fac*r[1];
    grad[2] = grad[2] + fac*r[2];
}

/* ---------------------------------------------------------------------------
    Isochrone potential
*/
double isochrone_value(double t, double *pars, double *q) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    double R2;
    R2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    return -pars[0] * pars[1] / (sqrt(R2 + pars[2]*pars[2]) + pars[2]);
}

void isochrone_gradient(double t, double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    double sqrt_r2_b2, fac, denom;
    sqrt_r2_b2 = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + pars[2]*pars[2]);
    denom = sqrt_r2_b2 * (sqrt_r2_b2 + pars[2])*(sqrt_r2_b2 + pars[2]);
    fac = pars[0] * pars[1] / denom;

    grad[0] = grad[0] + fac*r[0];
    grad[1] = grad[1] + fac*r[1];
    grad[2] = grad[2] + fac*r[2];
}

double isochrone_density(double t, double *pars, double *q) {
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

/* ---------------------------------------------------------------------------
    Hernquist sphere
*/
double hernquist_value(double t, double *pars, double *q) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    return -pars[0] * pars[1] / (R + pars[2]);
}

void hernquist_gradient(double t, double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R, fac;
    R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    fac = pars[0] * pars[1] / ((R + pars[2]) * (R + pars[2]) * R);

    grad[0] = grad[0] + fac*r[0];
    grad[1] = grad[1] + fac*r[1];
    grad[2] = grad[2] + fac*r[2];
}

double hernquist_density(double t, double *pars, double *q) {
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

/* ---------------------------------------------------------------------------
    Plummer sphere
*/
double plummer_value(double t, double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double R2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    return -pars[0]*pars[1] / sqrt(R2 + pars[2]*pars[2]);
}

void plummer_gradient(double t, double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double R2b, fac;
    R2b = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + pars[2]*pars[2];
    fac = pars[0] * pars[1] / sqrt(R2b) / R2b;

    grad[0] = grad[0] + fac*r[0];
    grad[1] = grad[1] + fac*r[1];
    grad[2] = grad[2] + fac*r[2];
}

double plummer_density(double t, double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    return 3*pars[1] / (4*M_PI*pars[2]*pars[2]*pars[2]) * pow(1 + r2/(pars[2]*pars[2]), -2.5);
}

/* ---------------------------------------------------------------------------
    Jaffe sphere

    TODO: I think this is all wrong?
*/
double jaffe_value(double t, double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R;
    R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    return pars[0] * pars[1] / pars[2] * log(R / (R + pars[2]));
}

void jaffe_gradient(double t, double *pars, double *r, double *grad){
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R, fac;
    R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    fac = pars[0] * pars[1] / ((R + pars[2]) * R * R);

    grad[0] = grad[0] + fac*r[0];
    grad[1] = grad[1] + fac*r[1];
    grad[2] = grad[2] + fac*r[2];
}

double jaffe_density(double t, double *pars, double *q) {
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
    Stone-Ostriker potential from Stone & Ostriker (2015)
*/
double stone_value(double t, double *pars, double *q) {
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

void stone_gradient(double t, double *pars, double *q, double *grad) {
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

double stone_density(double t, double *pars, double *q) {
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
double sphericalnfw_value(double t, double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (circular velocity at the scale radius)
            - r_s (scale radius)
    */
    double u, v_h2;
    v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) / pars[2];
    return -v_h2 * log(1 + u) / u;
}

void sphericalnfw_gradient(double t, double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (circular velocity at the scale radius)
            - r_s (scale radius)
    */
    double fac, u, v_h2;
    v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);

    u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) / pars[2];
    fac = v_h2 / (u*u*u) / (pars[2]*pars[2]) * (log(1+u) - u/(1+u));

    grad[0] = grad[0] + fac*r[0];
    grad[1] = grad[1] + fac*r[1];
    grad[2] = grad[2] + fac*r[2];
}

double sphericalnfw_density(double t, double *pars, double *q) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (circular velocity at the scale radius)
            - r_s (scale radius)
    */
    double v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    double r, rho0;
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);

    rho0 = v_h2 / (4*M_PI*pars[0]*pars[2]*pars[2]);
    return rho0 / ((r/pars[2]) * pow(1+r/pars[2],2));
}

/* ---------------------------------------------------------------------------
    Flattened NFW
*/
double flattenednfw_value(double t, double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (circular velocity at the scale radius)
            - r_s (scale radius)
            - q (flattening)
    */
    double u, v_h2;
    v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]/(pars[3]*pars[3])) / pars[2];
    return -v_h2 * log(1 + u) / u;
}

void flattenednfw_gradient(double t, double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (circular velocity at the scale radius)
            - r_s (scale radius)
            - q (flattening)
    */
    double fac, u, v_h2;
    v_h2 = pars[1]*pars[1] / (log(2.) - 0.5);
    u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]/(pars[3]*pars[3])) / pars[2];

    fac = v_h2 / (u*u*u) / (pars[2]*pars[2]) * (log(1+u) - u/(1+u));

    grad[0] = grad[0] + fac*r[0];
    grad[1] = grad[1] + fac*r[1];
    grad[2] = grad[2] + fac*r[2]/(pars[3]*pars[3]);
}

double flattenednfw_density(double t, double *pars, double *xyz) {
    /*  pars:
            - G (Gravitational constant)
            - v_c (circular velocity at the scale radius)
            - r_s (scale radius)
            - q (flattening)
    */
    double v = pars[1]*pars[1] / (log(2.) - 0.5);
    double s = pars[2];
    double q = pars[3];
    double x = xyz[0];
    double y = xyz[1];
    double z = xyz[2];

    return -((2*(s*s)*(v*v)*((-(pow(q,6)*pow((x*x) + (y*y),2)*
                 ((-1 + 2*(q*q))*(s*s) +
                   (-1 + 4*(q*q))*((x*x) + (y*y)))) -
              pow(q,4)*((x*x) + (y*y))*
               (2*(s*s) + 3*(1 + 2*(q*q))*((x*x) + (y*y)))*(z*z) +
              (q*q)*((-3 + 2*(q*q))*(s*s) - 9*((x*x) + (y*y)))*
               pow(z,4) + (-5 + 2*(q*q))*pow(z,6))/
            pow((q*q)*((s*s) + (x*x) + (y*y)) + (z*z),2) +
           ((q*q)*(-1 + 2*(q*q))*((x*x) + (y*y)) +
              (3 - 2*(q*q))*(z*z))*
            log(1 + ((x*x) + (y*y) + (z*z)/(q*q))/(s*s))))/
       (pow(q,4)*pow((x*x) + (y*y) + (z*z)/(q*q),3))) / (4*M_PI*pars[0]);
}

/* ---------------------------------------------------------------------------
    Miyamoto-Nagai flattened potential
*/
double miyamotonagai_value(double t, double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    double zd;
    zd = (pars[2] + sqrt(r[2]*r[2] + pars[3]*pars[3]));
    return -pars[0] * pars[1] / sqrt(r[0]*r[0] + r[1]*r[1] + zd*zd);
}

void miyamotonagai_gradient(double t, double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - a (length scale 1) TODO
            - b (length scale 2) TODO
    */
    double sqrtz, zd, fac;

    sqrtz = sqrt(r[2]*r[2] + pars[3]*pars[3]);
    zd = pars[2] + sqrtz;
    fac = pars[0]*pars[1] * pow(r[0]*r[0] + r[1]*r[1] + zd*zd, -1.5);

    grad[0] = grad[0] + fac*r[0];
    grad[1] = grad[1] + fac*r[1];
    grad[2] = grad[2] + fac*r[2] * (1. + pars[2] / sqrtz);
}

double miyamotonagai_density(double t, double *pars, double *q) {
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
    double numer = (b*b*M / (4*M_PI)) * (a*R2 + (a + 2*sqrt_zb)*(a + sqrt_zb)*(a + sqrt_zb));
    double denom = pow(R2 + (a + sqrt_zb)*(a + sqrt_zb), 2.5) * sqrt_zb*sqrt_zb*sqrt_zb;

    return numer/denom;
}

/* ---------------------------------------------------------------------------
    Lee-Suto triaxial NFW from Lee & Suto (2003)
*/
double leesuto_value(double t, double *pars, double *r) {
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

    x = r[0];
    y = r[1];
    z = r[2];

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

void leesuto_gradient(double t, double *pars, double *r, double *grad) {
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

    x = r[0];
    y = r[1];
    z = r[2];

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

double leesuto_density(double t, double *pars, double *r) {
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

    x = r[0];
    y = r[1];
    z = r[2];

    u = sqrt(x*x + y*y/b_a2 + z*z/c_a2) / pars[2];
    return v_h2 / (u * (1+u)*(1+u)) / (4.*M_PI*pars[2]*pars[2]*pars[0]);
}

/* ---------------------------------------------------------------------------
    Logarithmic (triaxial)
*/
double logarithmic_value(double t, double *pars, double *r) {
    /* pars[0] is G -- unused here */
    double x, y, z;

    x = r[0]*cos(pars[6]) + r[1]*sin(pars[6]);
    y = -r[0]*sin(pars[6]) + r[1]*cos(pars[6]);
    z = r[2];

    return 0.5*pars[1]*pars[1] * log(pars[2]*pars[2] + // scale radius
                                     x*x/(pars[3]*pars[3]) +
                                     y*y/(pars[4]*pars[4]) +
                                     z*z/(pars[5]*pars[5]));
}

void logarithmic_gradient(double t, double *pars, double *r, double *grad) {
    /* pars[0] is G -- unused here */
    double x, y, z, ax, ay, az, fac;

    x = r[0]*cos(pars[6]) + r[1]*sin(pars[6]);
    y = -r[0]*sin(pars[6]) + r[1]*cos(pars[6]);
    z = r[2];

    fac = pars[1]*pars[1] / (pars[2]*pars[2] + x*x/(pars[3]*pars[3]) + y*y/(pars[4]*pars[4]) + z*z/(pars[5]*pars[5]));
    ax = fac*x/(pars[3]*pars[3]);
    ay = fac*y/(pars[4]*pars[4]);
    az = fac*z/(pars[5]*pars[5]);

    grad[0] = grad[0] + (ax*cos(pars[6]) - ay*sin(pars[6]));
    grad[1] = grad[1] + (ax*sin(pars[6]) + ay*cos(pars[6]));
    grad[2] = grad[2] + az;
}
