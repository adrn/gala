#include <math.h>
#include "cpotential.h"


void apply_shift_rotate(double *qp_in, double *q0, double *R, int n_dim,
                        double *qp_out) {
    double tmp[n_dim];
    int j;

    // Shift to the specified origin
    for (j=0; j < n_dim; j++) {
        tmp[j] = qp_in[j] - q0[j];
    }

    // Apply rotation matrix
    // NOTE: elsewhere, we enforce that rotation matrix only works for
    // ndim=2 or ndim=3, so here we can assume that!
    if (n_dim == 3) {
        qp_out[0] = R[0] * tmp[0] + R[1] * tmp[1] + R[2] * tmp[2];
        qp_out[1] = R[3] * tmp[0] + R[4] * tmp[1] + R[5] * tmp[2];
        qp_out[2] = R[6] * tmp[0] + R[7] * tmp[1] + R[8] * tmp[2];
    } else if (n_dim == 2) {
        qp_out[0] = R[0] * tmp[0] + R[1] * tmp[1];
        qp_out[1] = R[2] * tmp[0] + R[3] * tmp[1];
    } else {
        for (j=0; j < n_dim; j++)
            qp_out[j] = tmp[j];
    }
}


double c_potential(CPotential *p, double t, double *qp) {
    double v = 0;
    int i;
    double qp_trans[p->n_dim];

    for (i=0; i < p->n_components; i++) {
        apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, &qp_trans[0]);
        v = v + (p->value)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim);
    }

    return v;
}


double c_density(CPotential *p, double t, double *qp) {
    double v = 0;
    int i;
    double qp_trans[p->n_dim];

    for (i=0; i < p->n_components; i++) {
        apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, &qp_trans[0]);
        v = v + (p->density)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim);
    }

    return v;
}


void c_gradient(CPotential *p, double t, double *qp, double *grad) {
    int i;
    double qp_trans[p->n_dim];

    for (i=0; i < p->n_dim; i++) {
        grad[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, &qp_trans[0]);
        (p->gradient)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim, grad);
    }

}


void c_hessian(CPotential *p, double t, double *qp, double *hess) {
    int i;
    double qp_trans[p->n_dim];

    for (i=0; i < pow(p->n_dim,2); i++) {
        hess[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, &qp_trans[0]);
        (p->hessian)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim, hess);
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


double c_mass_enclosed(CPotential *p, double t, double *qp, double G,
                       double *epsilon) {
    double r2, dPhi_dr;
    int j;

    r2 = 0;
    for (j=0; j<p->n_dim; j++) {
        r2 = r2 + qp[j]*qp[j];
    }
    dPhi_dr = c_d_dr(p, t, qp, epsilon);
    return fabs(r2 * dPhi_dr / G);
}
