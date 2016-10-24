#include <math.h>
#include "cpotential.h"
#include "cframe.h"

// TODO: Frame has to have parameters too

double c_value(CPotential *p, CFrame *f, double t, double *qp) {
    double v = 0;
    int i;

    for (i=0; i < p->n_components; i++) {
        v = v + (p->value)[i](t, (p->parameters)[i], qp);
    }

    v = v + (f->potential)(t, (f->parameters), qp);

    return v;
}

double c_density(CPotential *p, double t, double *qp) {
    double v = 0;
    int i;

    for (i=0; i < p->n_components; i++) {
        v = v + (p->density)[i](t, (p->parameters)[i], qp);
    }

    // TODO: I don't think a frame will ever contribute here...

    return v;
}

void c_gradient(CPotential *p, CFrame *f, double t, double *qp, double *grad) {
    int i;

    // TODO: grad, n_dim now have to be full phase-space dimensionality!!
    for (i=0; i < (p->n_dim); i++) {
        grad[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        (p->gradient)[i](t, (p->parameters)[i], qp, grad);
    }

    (f->gradient)(t, (f->parameters), qp, grad);

}

void c_hessian(CPotential *p, CFrame *f, double t, double *qp, double *hess) {
    int i;

    // TODO: hessian is now a bigger matrix...need to write to the correct submatrix in p->hessian
    for (i=0; i < pow(p->n_dim,2); i++) {
        hess[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        (p->hessian)[i](t, (p->parameters)[i], qp, hess);
    }

    // TODO: can I just add in the terms from the frame here?
    // (f->hessian)(t, (f->parameters), qp);
}

double c_d_dr(CPotential *p, CFrame *f, double t, double *qp, double *epsilon) {
    double h, r, dPhi_dr;
    int j;

    // TODO: allow user to specify fractional step-size
    h = 0.01;

    // Step-size for estimating radial gradient of the potential
    r = sqrt(qp[0]*qp[0] + qp[1]*qp[1] + qp[2]*qp[2]);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * qp[j]/r + qp[j];

    dPhi_dr = c_value(p, f, t, epsilon);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * qp[j]/r - qp[j];

    dPhi_dr = dPhi_dr - c_value(p, f, t, epsilon);

    return dPhi_dr / (2.*h);
}

double c_d2_dr2(CPotential *p, CFrame *f, double t, double *qp, double *epsilon) {
    double h, r, d2Phi_dr2;
    int j;

    // TODO: allow user to specify fractional step-size
    h = 0.01;

    // Step-size for estimating radial gradient of the potential
    r = sqrt(qp[0]*qp[0] + qp[1]*qp[1] + qp[2]*qp[2]);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * qp[j]/r + qp[j];
    d2Phi_dr2 = c_value(p, f, t, epsilon);

    d2Phi_dr2 = d2Phi_dr2 - 2.*c_value(p, f, t, qp);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * qp[j]/r - qp[j];
    d2Phi_dr2 = d2Phi_dr2 + c_value(p, f, t, epsilon);

    return d2Phi_dr2 / (h*h);
}

double c_mass_enclosed(CPotential *p, CFrame *f, double t, double *qp, double G, double *epsilon) {
    double r, dPhi_dr;
    r = sqrt(qp[0]*qp[0] + qp[1]*qp[1] + qp[2]*qp[2]);
    dPhi_dr = c_d_dr(p, f, t, qp, epsilon);
    return fabs(r*r * dPhi_dr / G);
}
