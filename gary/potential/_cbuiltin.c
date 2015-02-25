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
