#include <stddef.h>
#include <math.h>
#include "chamiltonian.h"
#include "potential/src/cpotential.h"
#include "frame/src/cframe.h"

double hamiltonian_value(CPotential *p, CFrameType *fr, double t, double *qp) {
    double v = 0;
    int i;

    v = v + (fr->energy)(t, (fr->parameters), qp, p->n_dim, NULL);

    for (i=0; i < p->n_components; i++) {
        // TODO: change potential 'value' -> 'energy'
        v = v + (p->value)[i](t, (p->parameters)[i], qp, p->n_dim, (p->state)[i]);
    }

    return v;
}

void hamiltonian_gradient(CPotential *p, CFrameType *fr, double t, double *qp, double *dH) {
    int i;

    for (i=0; i < 2*(p->n_dim); i++) {
        dH[i] = 0.;
    }

    // potential gradient has to be first
    c_gradient(p, 1, t, qp, &(dH[p->n_dim]));

    (fr->gradient)(t, (fr->parameters), qp, p->n_dim, 1, dH, NULL);

    for (i=p->n_dim; i < 2*(p->n_dim); i++) {
        dH[i] = -dH[i]; // pdot = -dH/dq
    }
}

void hamiltonian_gradient_T(CPotential *p, CFrameType *fr, size_t n, double t, double *qp_T, double *dH_T) {
    // qp_T: shape (n_dim, n)
    // dH_T: shape (n_dim, n)

    int ndim = p->n_dim;

    // Initialize dH_T to zeros
    for (int i = 0; i < 2 * ndim * n; i++) {
        dH_T[i] = 0.0;
    }

    // Call gradient functions directly with transposed data
    c_gradient(p, n, t, qp_T, dH_T + ndim * n);  // Write to momentum part
    (fr->gradient)(t, (fr->parameters), qp_T, ndim, n, dH_T, NULL);  // Write to position part

    // Negate the momentum derivatives
    for (int i = 0; i < n * ndim; i++) {
        dH_T[ndim * n + i] *= -1;  // pdot = -dH/dq
    }
}

void hamiltonian_hessian(CPotential *p, CFrameType *fr, double t, double *qp, double *d2H) {
    int i;

    for (i=0; i < p->n_components; i++) {
        (p->hessian)[i](t, (p->parameters)[i], qp, p->n_dim, d2H, (p->state)[i]);
    }

    // TODO: not implemented!!
    // TODO: can I just add in the terms from the frame here?
    // (fr->hessian)(t, (fr->parameters), qp, p->n_dim, d2H);
}
