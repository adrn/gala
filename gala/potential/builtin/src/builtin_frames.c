#include <math.h>
#include <string.h>

double static_frame_potential(double t, double *pars, double *qp) {
    /* No-op */
    return 0.;
}

void static_frame_gradient(double t, double *pars, double *qp, double *grad) {
    /* No-op */
}

void static_frame_hessian(double t, double *pars, double *qp, double *hess) {
    /* No-op */
}
