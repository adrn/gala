#include <math.h>
#include "cpotential.h"


void apply_rotate(double *q_in, double *R, int n_dim, int transpose,
                  double *q_out) {
    // NOTE: elsewhere, we enforce that rotation matrix only works for
    // ndim=2 or ndim=3, so here we can assume that!
    if (n_dim == 3) {
        if (transpose == 0) {
            q_out[0] = q_out[0] + R[0] * q_in[0] + R[1] * q_in[1] + R[2] * q_in[2];
            q_out[1] = q_out[1] + R[3] * q_in[0] + R[4] * q_in[1] + R[5] * q_in[2];
            q_out[2] = q_out[2] + R[6] * q_in[0] + R[7] * q_in[1] + R[8] * q_in[2];
        } else {
            q_out[0] = q_out[0] + R[0] * q_in[0] + R[3] * q_in[1] + R[6] * q_in[2];
            q_out[1] = q_out[1] + R[1] * q_in[0] + R[4] * q_in[1] + R[7] * q_in[2];
            q_out[2] = q_out[2] + R[2] * q_in[0] + R[5] * q_in[1] + R[8] * q_in[2];
        }
    } else if (n_dim == 2) {
        if (transpose == 0) {
            q_out[0] = q_out[0] + R[0] * q_in[0] + R[1] * q_in[1];
            q_out[1] = q_out[1] + R[2] * q_in[0] + R[3] * q_in[1];
        } else {
            q_out[0] = q_out[0] + R[0] * q_in[0] + R[2] * q_in[1];
            q_out[1] = q_out[1] + R[1] * q_in[0] + R[3] * q_in[1];
        }
    } else {
        for (int j=0; j < n_dim; j++)
            q_out[j] = q_out[j] + q_in[j];
    }
}


void apply_shift_rotate(double *q_in, double *q0, double *R, int n_dim,
                        int transpose, double *q_out) {
    double tmp[n_dim];
    int j;

    // Shift to the specified origin
    for (j=0; j < n_dim; j++) {
        tmp[j] = q_in[j] - q0[j];
    }

    // Apply rotation matrix
    apply_rotate(&tmp[0], R, n_dim, transpose, q_out);
}


double c_potential(CPotential *p, double t, double *qp) {
    double v = 0;
    int i, j;
    double qp_trans[p->n_dim];

    for (i=0; i < p->n_components; i++) {
        for (j=0; j < p->n_dim; j++)
            qp_trans[j] = 0.;
        apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, 0,
                           &qp_trans[0]);
        v = v + (p->value)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim);
    }

    return v;
}


double c_density(CPotential *p, double t, double *qp) {
    double v = 0;
    int i, j;
    double qp_trans[p->n_dim];

    for (i=0; i < p->n_components; i++) {
        for (j=0; j < p->n_dim; j++)
            qp_trans[j] = 0.;
        apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, 0,
                           &qp_trans[0]);
        v = v + (p->density)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim);
    }

    return v;
}


void c_gradient(CPotential *p, double t, double *qp, double *grad) {
    int i, j;
    double qp_trans[p->n_dim];
    double tmp_grad[p->n_dim];

    for (i=0; i < p->n_dim; i++) {
        grad[i] = 0.;
        tmp_grad[i] = 0.;
        qp_trans[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        for (j=0; j < p->n_dim; j++) {
            tmp_grad[j] = 0.;
            qp_trans[j] = 0.;
        }

        apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, 0,
                           &qp_trans[0]);
        (p->gradient)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim,
                         &tmp_grad[0]);
        apply_rotate(&tmp_grad[0], (p->R)[i], p->n_dim, 1, &grad[0]);
    }
}


void c_hessian(CPotential *p, double t, double *qp, double *hess) {
    int i;
    double qp_trans[p->n_dim];

    for (i=0; i < pow(p->n_dim,2); i++) {
        hess[i] = 0.;

        if (i < p->n_dim) {
            qp_trans[i] = 0.;
        }
    }

    for (i=0; i < p->n_components; i++) {
        apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, 0,
                           &qp_trans[0]);
        (p->hessian)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim, hess);
        // TODO: here - need to apply inverse rotation to the Hessian!
        // - Hessian calculation for potentials with rotations are disabled
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
    h = 1E-4;

    // Step-size for estimating radial gradient of the potential
    r = sqrt(r2);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = qp[j] + h * qp[j]/r;

    dPhi_dr = c_potential(p, t, epsilon);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = qp[j] - h * qp[j]/r;

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
    h = 1E-2;

    // Step-size for estimating radial gradient of the potential
    r = sqrt(r2);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = qp[j] + h * qp[j]/r;
    d2Phi_dr2 = c_potential(p, t, epsilon);

    d2Phi_dr2 = d2Phi_dr2 - 2.*c_potential(p, t, qp);

    for (j=0; j < (p->n_dim); j++)
        epsilon[j] = qp[j] - h * qp[j]/r;
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


// TODO: This isn't really the right place for this...
void c_nbody_acceleration(CPotential **pots, double t, double *qp,
                          int norbits, int nbody, int ndim, double *acc) {
    int i, j, k;
    CPotential *body_pot;
    int ps_ndim = 2 * ndim; // 6, for 3D position/velocity
    double f2[ndim];

    for (j=0; j < nbody; j++) { // the particles generating force
        body_pot = pots[j];

        if ((body_pot->null) == 1)
            continue;

        for (i=0; i < body_pot->n_components; i++)
            (body_pot->q0)[i] = &qp[j * ps_ndim];

        for (i=0; i < norbits; i++) {
            if (i != j) {
                c_gradient(body_pot, t, &qp[i * ps_ndim], &f2[0]);
                for (k=0; k < ndim; k++)
                   acc[i*ps_ndim + ndim + k] += -f2[k];
            }
        }
    }
}

// TODO: this is a hack to get nbody leapfrog working
void c_nbody_gradient_symplectic(
    CPotential **pots, double t, double *w,
    double *nbody_w, int nbody, int nbody_i,
    int ndim, double *grad
) {
    int i, j, k;
    CPotential *body_pot;
    double f2[ndim];

    for (j=0; j < nbody; j++) { // the particles generating force
        body_pot = pots[j];

        if ((body_pot->null == 1) || (j == nbody_i))
            continue;

        for (i=0; i < body_pot->n_components; i++) {
            (body_pot->q0)[i] = &nbody_w[j * 2 * ndim]; // p-s ndim
        }

        c_gradient(body_pot, t, w, &f2[0]);
        for (k=0; k < ndim; k++)
            grad[k] += f2[k];
    }
}