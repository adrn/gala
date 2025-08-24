#include <math.h>
#include <stdlib.h>
#include "cpotential.h"
#include "src/vectorization.h"

CPotential* allocate_cpotential(int n_components) {
    CPotential* p = (CPotential*)malloc(sizeof(CPotential));

    p->n_components = n_components;
    p->n_dim = 0;
    p->null = 0;

    // Allocate arrays
    p->density = (densityfunc*)malloc(n_components * sizeof(densityfunc));
    p->value = (energyfunc*)malloc(n_components * sizeof(energyfunc));
    p->gradient = (gradientfunc*)malloc(n_components * sizeof(gradientfunc));
    p->hessian = (hessianfunc*)malloc(n_components * sizeof(hessianfunc));
    p->n_params = (int*)malloc(n_components * sizeof(int));
    p->parameters = (double**)malloc(n_components * sizeof(double*));
    p->q0 = (double**)malloc(n_components * sizeof(double*));
    p->R = (double**)malloc(n_components * sizeof(double*));
    p->state = (void**)malloc(n_components * sizeof(void*));
    p->do_shift_rotate = (int*)malloc(n_components * sizeof(int));

        // Initialize with NULL pointers
        for (int i = 0; i < n_components; i++) {
            p->parameters[i] = NULL;
            p->q0[i] = NULL;
            p->R[i] = NULL;
            p->state[i] = NULL;
        }

        return p;
    }

    void free_cpotential(CPotential* p) {
        if (p == NULL) return;

        free(p->density);
        free(p->value);
        free(p->gradient);
        free(p->hessian);
        free(p->n_params);
        free(p->parameters); // Note: doesn't free the actual parameter arrays
        free(p->q0);         // Note: doesn't free the actual q0 arrays
        free(p->R);          // Note: doesn't free the actual R arrays
        free(p->state);          // Note: doesn't free the actual state arrays
        free(p->do_shift_rotate);
        free(p);
    }

    int resize_cpotential_arrays(CPotential* pot, int new_n_components) {
        if (new_n_components <= pot->n_components)
            return 1;  // Nothing to do

        // Reallocate arrays to the new size
        pot->n_components = new_n_components;
        pot->density = (densityfunc*)realloc(pot->density, new_n_components * sizeof(densityfunc));
        pot->value = (energyfunc*)realloc(pot->value, new_n_components * sizeof(energyfunc));
        pot->gradient = (gradientfunc*)realloc(pot->gradient, new_n_components * sizeof(gradientfunc));
        pot->hessian = (hessianfunc*)realloc(pot->hessian, new_n_components * sizeof(hessianfunc));
        pot->n_params = (int*)realloc(pot->n_params, new_n_components * sizeof(int));
        pot->parameters = (double**)realloc(pot->parameters, new_n_components * sizeof(double*));
        pot->q0 = (double**)realloc(pot->q0, new_n_components * sizeof(double*));
        pot->R = (double**)realloc(pot->R, new_n_components * sizeof(double*));
        pot->state = (void**)realloc(pot->state, new_n_components * sizeof(void*));
        pot->do_shift_rotate = (int*)realloc(pot->do_shift_rotate, new_n_components * sizeof(int));

        // Initialize new elements
        for (int i = pot->n_components; i < new_n_components; i++) {
            pot->parameters[i] = NULL;
            pot->q0[i] = NULL;
            pot->R[i] = NULL;
            pot->state[i] = NULL;
        }

        return 1;  // Success
    }

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

void apply_rotate_T(double6ptr q, const double *R, int n_dim, int transpose) {
    // in-place rotation

    double x = q[0];
    if (n_dim == 3) {
        double y = q[1];
        double z = q[2];
        if (transpose == 0) {
            q[0] = R[0] * x + R[1] * y + R[2] * z;
            q[1] = R[3] * x + R[4] * y + R[5] * z;
            q[2] = R[6] * x + R[7] * y + R[8] * z;
        } else {
            q[0] = R[0] * x + R[3] * y + R[6] * z;
            q[1] = R[1] * x + R[4] * y + R[7] * z;
            q[2] = R[2] * x + R[5] * y + R[8] * z;
        }
    } else if (n_dim == 2) {
        double y = q[1];
        if (transpose == 0) {
            q[0] = R[0] * x + R[1] * y;
            q[1] = R[2] * x + R[3] * y;
        } else {
            q[0] = R[0] * x + R[2] * y;
            q[1] = R[1] * x + R[3] * y;
        }
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

void apply_shift_rotate_N(size_t N, const double *q_in, const double *q0, const double *R, int n_dim,
                        int transpose, double *q_out) {
    // q_in: shape [n_dim, N]
    // q_out: shape [n_dim, N]

    for(size_t i = 0; i < N; i++) {
        // Shift to the specified origin
        for (int j=0; j < n_dim; j++) {
            q_out[j * N + i] = q_in[j * N + i] - q0[j];
        }

        // Apply rotation matrix in place
        apply_rotate_T(
            double6ptr{q_out + i, N}, R, n_dim, transpose
        );
    }
}


double c_potential(CPotential *p, double t, double *qp) {
    double v = 0;
    int i, j;
    double qp_trans[p->n_dim];

    for (i=0; i < p->n_components; i++) {
        if (p->do_shift_rotate[i] == 0) {
            v = v + (p->value)[i](t, (p->parameters)[i], &qp[0], p->n_dim, (p->state)[i]);
            continue;
        } else {
            for (j=0; j < p->n_dim; j++)
                qp_trans[j] = 0.;
            apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, 0,
                            &qp_trans[0]);
            v = v + (p->value)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim, (p->state)[i]);
        }
    }

    return v;
}


double c_density(CPotential *p, double t, double *qp) {
    double v = 0;
    int i, j;
    double qp_trans[p->n_dim];

    for (i=0; i < p->n_components; i++) {
        if (p->do_shift_rotate[i] == 0) {
            v = v + (p->density)[i](t, (p->parameters)[i], &qp[0], p->n_dim, (p->state)[i]);
            continue;
        } else {
            for (j=0; j < p->n_dim; j++)
                qp_trans[j] = 0.;
            apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, 0,
                            &qp_trans[0]);
            v = v + (p->density)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim, (p->state)[i]);
        }
    }

    return v;
}


void c_gradient(CPotential *p, size_t N, double t, double *qp, double *grad) {
    // qp: shape [p->n_dim, N]
    // grad: shape [p->n_dim, N]

    double *qp_trans = NULL;
    double *tmp_grad = NULL;
    bool need_transform = false;

    // Check if any components need transformation
    for (size_t i = 0; i < p->n_components; i++) {
        if (p->do_shift_rotate[i] != 0) {
            need_transform = true;
            break;
        }
    }

    // Allocate temporary arrays if transformation is needed
    if (need_transform) {
        qp_trans = (double*)malloc(p->n_dim * N * sizeof(double));
        tmp_grad = (double*)malloc(p->n_dim * N * sizeof(double));
    }

    // Initialize gradient array
    // TODO: may need to remove this for n-body-style accumulations
    for (size_t i = 0; i < p->n_dim * N; i++) {
        grad[i] = 0.;
    }

    for (size_t i = 0; i < p->n_components; i++) {
        if (p->do_shift_rotate[i] == 0) {
            (p->gradient)[i](N, t, (p->parameters)[i], qp, p->n_dim, grad, (p->state)[i]);
        } else {
            // Initialize temporary arrays
            for (size_t j = 0; j < p->n_dim * N; j++) {
                qp_trans[j] = 0.;
                tmp_grad[j] = 0.;
            }

            // Apply shift and rotation to all particles
            apply_shift_rotate_N(
                N,
                qp,
                (p->q0)[i],
                (p->R)[i],
                p->n_dim,
                0,
                qp_trans
            );

            // Compute gradient for transformed coordinates
            (p->gradient)[i](N, t, (p->parameters)[i], qp_trans, p->n_dim, tmp_grad, (p->state)[i]);

            // Apply inverse rotation to gradient and accumulate
            for (size_t j = 0; j < N; j++) {
                apply_rotate_T(
                    double6ptr{tmp_grad + j, N},
                    (p->R)[i],
                    p->n_dim,
                    1
                );
            }

            for (size_t j = 0; j < p->n_dim * N; j++) {
                grad[j] += tmp_grad[j];
            }
        }
    }

    // Free temporary arrays
    if (need_transform) {
        free(qp_trans);
        free(tmp_grad);
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
        if (p->do_shift_rotate[i] == 0) {
            (p->hessian)[i](t, (p->parameters)[i], &qp[0], p->n_dim, hess, (p->state)[i]);
            continue;
        } else {
            apply_shift_rotate(qp, (p->q0)[i], (p->R)[i], p->n_dim, 0, &qp_trans[0]);
            (p->hessian)[i](t, (p->parameters)[i], &qp_trans[0], p->n_dim, hess, (p->state)[i]);
            // TODO: here - need to apply inverse rotation to the Hessian!
            // - Hessian calculation for potentials with rotations are disabled
        }
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

        for (i=0; i < body_pot->n_components; i++) {
            (body_pot->do_shift_rotate)[i] = 1;
            (body_pot->q0)[i] = &qp[j * ps_ndim];
        }

        for (i=0; i < norbits; i++) {
            if (i != j) {
                c_gradient(body_pot, 1, t, &qp[i * ps_ndim], &f2[0]);
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
            (body_pot->do_shift_rotate)[i] = 1;
            (body_pot->q0)[i] = &nbody_w[j * 2 * ndim]; // p-s ndim
        }

        c_gradient(body_pot, 1, t, w, &f2[0]);
        for (k=0; k < ndim; k++)
            grad[k] += f2[k];
    }
}
