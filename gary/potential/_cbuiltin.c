#include <math.h>

/* ---------------------------------------------------------------------------
    Kepler potential
*/
double kepler_value(double *pars, double *q) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    double R;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    return -pars[0] * pars[1] / R;
}

void kepler_gradient(double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
    */
    double R, fac;
    R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    fac = pars[0] * pars[1] / (R*R*R);

    grad[0] = fac*r[0];
    grad[1] = fac*r[1];
    grad[2] = fac*r[2];
}

/* ---------------------------------------------------------------------------
    Isochrone potential
*/
double isochrone_value(double *pars, double *q) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    double R2;
    R2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    return -pars[0] * pars[1] / (sqrt(R2 + pars[2]*pars[2]) + pars[2]);
}

void isochrone_gradient(double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (core scale)
    */
    double sqrtR2b, fac, denom;
    sqrtR2b = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + pars[2]*pars[2]);
    denom = (sqrtR2b + pars[2]*pars[2]);
    fac = pars[0] * pars[1] / (denom * denom * sqrtR2b);

    grad[0] = fac*r[0];
    grad[1] = fac*r[1];
    grad[2] = fac*r[2];
}

/* ---------------------------------------------------------------------------
    Hernquist sphere
*/
double hernquist_value(double *pars, double *q) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    return -pars[0] * pars[1] / (R + pars[2]);
}

void hernquist_gradient(double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R, fac;
    R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    fac = pars[0] * pars[1] / ((R + pars[2]) * (R + pars[2]) * R);

    grad[0] = fac*r[0];
    grad[1] = fac*r[1];
    grad[2] = fac*r[2];
}

/* ---------------------------------------------------------------------------
    Plummer sphere
*/
double plummer_value(double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double R2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
    return -pars[0]*pars[1] / sqrt(R2 + pars[2]*pars[2]);
}

void plummer_gradient(double *pars, double *r, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - b (length scale)
    */
    double R2b, fac;
    R2b = r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + pars[2]*pars[2];
    fac = pars[0] * pars[1] / sqrt(R2b) / R2b;

    grad[0] = fac*r[0];
    grad[1] = fac*r[1];
    grad[2] = fac*r[2];
}

/* ---------------------------------------------------------------------------
    Jaffe sphere
*/
double jaffe_value(double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R;
    R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    return pars[0] * pars[1] / pars[2] * log(R / (R + pars[2]));
}

void jaffe_gradient(double *pars, double *r, double *grad){
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R, fac;
    R = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    fac = pars[0] * pars[1] / ((R + pars[2]) * R * R);

    grad[0] = fac*r[0];
    grad[1] = fac*r[1];
    grad[2] = fac*r[2];
}

/* ---------------------------------------------------------------------------
    Stone-Ostriker potential from Stone & Ostriker (2015)
*/
double stone_value(double *pars, double *r) {
    /*  pars:
            - G (Gravitational constant)
            - m (total mass)
            - r_c (core radius)
            - r_t (truncation radius)
    */
    double rr, u_c, u_t, f;
    rr = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    u_c = rr / pars[2];
    u_t = rr / pars[3];
    f = M_PI * (pars[3]*pars[3] - pars[2]*pars[2]) / (pars[2] + pars[3]);
    return -pars[0] * pars[1] / f * (atan(u_t)/u_t - atan(u_c)/u_c +
                      0.5*log((rr*rr + pars[3]*pars[3])/(rr*rr + pars[2]*pars[2])));
}

void stone_gradient(double *pars, double *r, double *grad) {
    double dphi_dr, rr;
    // TODO: not implemented

    grad[0] = dphi_dr*r[0]/rr;
    grad[1] = dphi_dr*r[1]/rr;
    grad[2] = dphi_dr*r[2]/rr;
}

/* ---------------------------------------------------------------------------
    Spherical NFW
*/
double sphericalnfw_value(double *pars, double *r) {
    double u, v_h2;
    v_h2 = pars[0]*pars[0] / (log(2.) - 0.5);
    u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) / pars[1];
    return -v_h2 * log(1 + u) / u;
}

void sphericalnfw_gradient(double *pars, double *r, double *grad) {
    double fac, u, v_h2;
    v_h2 = pars[0]*pars[0] / (log(2.) - 0.5);

    u = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) / pars[1];
    fac = v_h2 / (u*u*u) / (pars[1]*pars[1]) * (log(1+u) - u/(1+u));

    grad[0] = fac*r[0];
    grad[1] = fac*r[1];
    grad[2] = fac*r[2];
}

/* ---------------------------------------------------------------------------
    Miyamoto-Nagai flattened potential
*/
double miyamotonagai_value(double *pars, double *r) {
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

void miyamotonagai_gradient(double *pars, double *r, double *grad) {
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

    grad[0] = fac*r[0];
    grad[1] = fac*r[1];
    grad[2] = fac*r[2] * (1. + pars[2] / sqrtz);
}

/* ---------------------------------------------------------------------------
    Lee-Suto triaxial NFW from Lee & Suto (2003)
*/
double leesuto_value(double *pars, double *r) {
    /*  pars: TODO
            -
    */
    double x, y, z, _r, u, v_h2;
    double e_b2 = 1-pow(pars[3]/pars[2],2);
    double e_c2 = 1-pow(pars[4]/pars[2],2);

    v_h2 = pars[0]*pars[0] / (log(2.) - 0.5 + (log(2.)-0.75)*e_b2 + (log(2.)-0.75)*e_c2);

    // pars[5] up to and including pars[10] are R
    x = pars[5]*r[0]  + pars[6]*r[1]  + pars[7]*r[2];
    y = pars[8]*r[0]  + pars[9]*r[1]  + pars[10]*r[2];
    z = pars[11]*r[0] + pars[12]*r[1] + pars[13]*r[2];

    _r = sqrt(x*x + y*y + z*z);
    u = _r / pars[1];
    return v_h2 * ((e_b2/2 + e_c2/2)*((1/u - 1/(u*u*u))*log(u + 1) - 1 + (2*u*u - 3*u + 6)/(6*u*u)) + (e_b2*y*y/(2*_r*_r) + e_c2*z*z/(2*_r*_r))*((u*u - 3*u - 6)/(2*u*u*(u + 1)) + 3*log(u + 1)/(u*u*u)) - log(u + 1)/u);
}

void leesuto_gradient(double *pars, double *r, double *grad) {
    /*  pars: TODO
            -
    */
    double x, y, z, _r, _r2, _r4, ax, ay, az;
    double v_h2, x0, x2, x22;
    double x20, x21, x7, x1;
    double x10, x13, x15, x16, x17;
    double e_b2 = 1-pow(pars[3]/pars[2],2);
    double e_c2 = 1-pow(pars[4]/pars[2],2);

    v_h2 = pars[0]*pars[0] / (log(2.) - 0.5 + (log(2.)-0.75)*e_b2 + (log(2.)-0.75)*e_c2);

    // pars[5] up to and including pars[10] are R
    x = pars[5]*r[0]  + pars[6]*r[1]  + pars[7]*r[2];
    y = pars[8]*r[0]  + pars[9]*r[1]  + pars[10]*r[2];
    z = pars[11]*r[0] + pars[12]*r[1] + pars[13]*r[2];

    _r2 = x*x + y*y + z*z;
    _r = sqrt(_r2);
    _r4 = _r2*_r2;

    x0 = _r + pars[1];
    x1 = x0*x0;
    x2 = v_h2/(12.*_r4*_r2*_r*x1);
    x10 = log(x0/pars[1]);

    x13 = _r*3.*pars[1];
    x15 = x13 - _r2;
    x16 = x15 + 6.*(pars[1]*pars[1]);
    x17 = 6.*pars[1]*x0*(_r*x16 - x0*x10*6.*(pars[1]*pars[1]));
    x20 = x0*_r2;
    x21 = 2.*_r*x0;
    x7 = e_b2*y*y + e_c2*z*z;
    x22 = -12.*_r4*_r*pars[1]*x0 + 12.*_r4*pars[1]*x1*x10 + 3.*pars[1]*x7*(x16*_r2 - 18.*x1*x10*(pars[1]*pars[1]) + x20*(2.*_r - 3.*pars[1]) + x21*(x15 + 9.*(pars[1]*pars[1]))) - x20*(e_b2 + e_c2)*(-6.*_r*pars[1]*(_r2 - (pars[1]*pars[1])) + 6.*pars[1]*x0*x10*(_r2 - 3.*(pars[1]*pars[1])) + x20*(-4.*_r + 3.*pars[1]) + x21*(-x13 + 2.*_r2 + 6.*(pars[1]*pars[1])));

    ax = x2*x*(x17*x7 + x22);
    ay = x2*y*(x17*(x7 - _r2*e_b2) + x22);
    az = x2*z*(x17*(x7 - _r2*e_c2) + x22);

    grad[0] = pars[5]*ax  + pars[8]*ay  + pars[11]*az;
    grad[1] = pars[6]*ax  + pars[9]*ay  + pars[12]*az;
    grad[2] = pars[7]*ax  + pars[10]*ay + pars[13]*az;
}

/* ---------------------------------------------------------------------------
    Logarithmic (triaxial)
*/
double logarithmic_value(double *pars, double *r) {
    double x, y, z;

    // pars[5] up to and including pars[10] are R
    x = pars[5]*r[0]  + pars[6]*r[1]  + pars[7]*r[2];
    y = pars[8]*r[0]  + pars[9]*r[1]  + pars[10]*r[2];
    z = pars[11]*r[0] + pars[12]*r[1] + pars[13]*r[2];

    return 0.5*pars[0]*pars[0] * log(pars[1]*pars[1] + // scale radius
                                     x*x/(pars[2]*pars[2]) +
                                     y*y/(pars[3]*pars[3]) +
                                     z*z/(pars[4]*pars[4]));
}

void logarithmic_gradient(double *pars, double *r, double *grad) {

    double x, y, z, _r, _r2, ax, ay, az, fac;

    // pars[5] up to and including pars[10] are R
    x = pars[5]*r[0]  + pars[6]*r[1]  + pars[7]*r[2];
    y = pars[8]*r[0]  + pars[9]*r[1]  + pars[10]*r[2];
    z = pars[11]*r[0] + pars[12]*r[1] + pars[13]*r[2];

    _r2 = x*x + y*y + z*z;
    _r = sqrt(_r2);

    fac = pars[0]*pars[0] / (pars[1]*pars[1] + x*x/(pars[2]*pars[2]) + y*y/(pars[3]*pars[3]) + z*z/(pars[4]*pars[4]));
    ax = fac*x/(pars[2]*pars[2]);
    ay = fac*y/(pars[3]*pars[3]);
    az = fac*z/(pars[4]*pars[4]);

    grad[0] = pars[5]*ax  + pars[8]*ay  + pars[11]*az;
    grad[1] = pars[6]*ax  + pars[9]*ay  + pars[12]*az;
    grad[2] = pars[7]*ax  + pars[10]*ay + pars[13]*az;
}

/* TOTAL HACK */
double lm10_value(double *pars, double*r) {
    double v = 0.;
    v += hernquist_value(&pars[0], &r[0]);
    v += miyamotonagai_value(&pars[3], &r[0]);
    v += logarithmic_value(&pars[7], &r[0]);
    return v;
}

void lm10_gradient(double *pars, double *r, double *grad) {
    double tmp_grad[3];
    int i;

    hernquist_gradient(&pars[0], &r[0], &tmp_grad[0]);
    for (i=0; i<3; i++) grad[i] = tmp_grad[i];

    miyamotonagai_gradient(&pars[3], &r[0], &tmp_grad[0]);
    for (i=0; i<3; i++) grad[i] += tmp_grad[i];

    logarithmic_gradient(&pars[7], &r[0], &tmp_grad[0]);
    for (i=0; i<3; i++) grad[i] += tmp_grad[i];
}
