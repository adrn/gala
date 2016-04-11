#include <math.h>
#include "cpotential.h"

double c_value(CPotential *p, double t, double *q) {
    double v = 0;
    for (int i=0; i < p->n_components; i++) {
        v = v + (p->value)[i](t, (p->parameters)[i], q);
    }
    return v;
}

double c_density(CPotential *p, double t, double *q) {
    double v = 0;
    for (int i=0; i < p->n_components; i++) {
        v = v + (p->density)[i](t, (p->parameters)[i], q);
    }
    return v;
}

void c_gradient(CPotential *p, double t, double *q, double *grad) {
    int i;

    for (i=0; i < (p->n_dim); i++) {
        grad[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        (p->gradient)[i](t, (p->parameters)[i], q, grad);
    }
}

double c_d_dr(CPotential *p, double t, double *q, double *epsilon) {
    double h, r, dPhi_dr;
    int j;

    // TODO: allow user to specify fractional step-size
    h = 0.01;

    // Step-size for estimating radial gradient of the potential
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * q[j]/r + q[j];

    dPhi_dr = c_value(p, t, epsilon);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * q[j]/r - q[j];

    dPhi_dr = dPhi_dr - c_value(p, t, epsilon);

    return dPhi_dr / (2.*h);
}

double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) {
    double h, r, d2Phi_dr2;
    int j;

    // TODO: allow user to specify fractional step-size
    h = 0.01;

    // Step-size for estimating radial gradient of the potential
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * q[j]/r + q[j];
    d2Phi_dr2 = c_value(p, t, epsilon);

    d2Phi_dr2 = d2Phi_dr2 - 2.*c_value(p, t, q);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = h * q[j]/r - q[j];
    d2Phi_dr2 = d2Phi_dr2 + c_value(p, t, epsilon);

    return d2Phi_dr2 / (h*h);
}

double c_mass_enclosed(CPotential *p, double t, double *q, double G, double *epsilon) {
    double r, dPhi_dr;
    r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
    dPhi_dr = c_d_dr(p, t, q, epsilon);
    return fabs(r*r * dPhi_dr / G);
}
