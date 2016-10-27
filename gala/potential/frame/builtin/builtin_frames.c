#include <math.h>
#include <string.h>

double static_frame_hamiltonian(double t, double *pars, double *qp) {
    int n_dim = 3; // TODO: HACK: damn, how do I get the n_dim from potential in here
    int i;
    double E = 0.;

    for (i=0; i<n_dim; i++) {
        E += qp[i+n_dim]*qp[i+n_dim]; // p^2
    }
    return 0.5*E; // kinetic term
}

void static_frame_gradient(double t, double *pars, double *qp, double *dH) {
    int n_dim = 3; // TODO: HACK: damn, how do I get the n_dim from potential in here
    int i;

    for (i=0; i < n_dim; i++) {
        dH[i] = qp[n_dim+i]; // qdot = p
    }
}

void static_frame_hessian(double t, double *pars, double *qp, double *d2H) {
    /* TODO: */
}
