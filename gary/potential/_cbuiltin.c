#include <math.h>

/*
    Hernquist spheroid potential from Hernquist (1990)
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

void hernquist_gradient(double *pars, double *q, double *grad) {
    /*  pars:
            - G (Gravitational constant)
            - m (mass scale)
            - c (length scale)
    */
    double R, fac;
    R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    fac = pars[0] * pars[1] / ((R + pars[2]) * (R + pars[2]) * R);

    grad[0] = fac*q[0];
    grad[1] = fac*q[1];
    grad[2] = fac*q[2];
}
