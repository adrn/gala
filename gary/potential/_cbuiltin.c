#include <math.h>

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
