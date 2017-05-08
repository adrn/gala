#include <math.h>
#include "cpotential.h"

double c_potential(CPotential *p, double t, double *qp) {
    double v = 0;
    int i, j;
    double _qp[p->n_dim];

    for (i=0; i < p->n_components; i++) {
        // this looks like it sucks, but doesn't add much time...
        for (j=0; j < p->n_dim; j++) {
            _qp[j] = qp[j] - (p->q0)[i][j];
        }
        v = v + (p->value)[i](t, (p->parameters)[i], &_qp[0], p->n_dim);
    }

    return v;
}

double c_density(CPotential *p, double t, double *qp) {
    double v = 0;
    int i, j;
    double _qp[p->n_dim];

    for (i=0; i < p->n_components; i++) {
        // this looks like it sucks, but doesn't add much time...
        for (j=0; j < p->n_dim; j++) {
            _qp[j] = qp[j] - (p->q0)[i][j];
        }
        v = v + (p->density)[i](t, (p->parameters)[i], &_qp[0], p->n_dim);
    }

    return v;
}

void c_gradient(CPotential *p, double t, double *qp, double *grad) {
    int i, j;
    double _qp[p->n_dim];

    for (i=0; i < p->n_dim; i++) {
        grad[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        // this looks like it sucks, but doesn't add much time...
        for (j=0; j < p->n_dim; j++) {
            _qp[j] = qp[j] - (p->q0)[i][j];
        }
        (p->gradient)[i](t, (p->parameters)[i], &_qp[0], p->n_dim, grad);
    }

}

void c_hessian(CPotential *p, double t, double *qp, double *hess) {
    int i, j;
    double _qp[p->n_dim];

    for (i=0; i < pow(p->n_dim,2); i++) {
        hess[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        // this looks like it sucks, but doesn't add much time...
        for (j=0; j < p->n_dim; j++) {
            _qp[j] = qp[j] - (p->q0)[i][j];
        }
        (p->hessian)[i](t, (p->parameters)[i], &_qp[0], p->n_dim, hess);
    }

}

double c_d_dr(CPotential *p, double t, double *qp, double *epsilon) {
    double h, r, dPhi_dr;
    int j;
    double r2 = 0;
    for (j=0; j<p->n_dim; j++) {
        r2 = r2 + qp[j]*qp[j];
    }

    // TODO: allow user to specify fractional step-size
    h = 1E-5;

    // Step-size for estimating radial gradient of the potential
    r = sqrt(r2);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * qp[j]/r + qp[j];

    dPhi_dr = c_potential(p, t, epsilon);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * qp[j]/r - qp[j];

    dPhi_dr = dPhi_dr - c_potential(p, t, epsilon);

    return dPhi_dr / (2.*h);
}

double c_d2_dr2(CPotential *p, double t, double *qp, double *epsilon) {
    double h, r, d2Phi_dr2;
    int j;
    double r2 = 0;
    for (j=0; j<p->n_dim; j++) {
        r2 = r2 + qp[j]*qp[j];
    }

    // TODO: allow user to specify fractional step-size
    h = 1E-5;

    // Step-size for estimating radial gradient of the potential
    r = sqrt(r2);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * qp[j]/r + qp[j];
    d2Phi_dr2 = c_potential(p, t, epsilon);

    d2Phi_dr2 = d2Phi_dr2 - 2.*c_potential(p, t, qp);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * qp[j]/r - qp[j];
    d2Phi_dr2 = d2Phi_dr2 + c_potential(p, t, epsilon);

    return d2Phi_dr2 / (h*h);
}

double c_mass_enclosed(CPotential *p, double t, double *qp, double G, double *epsilon) {
    double r2, dPhi_dr;
    int j;

    r2 = 0;
    for (j=0; j<p->n_dim; j++) {
        r2 = r2 + qp[j]*qp[j];
    }
    dPhi_dr = c_d_dr(p, t, qp, epsilon);
    return fabs(r2 * dPhi_dr / G);
}
