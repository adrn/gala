#include <math.h>
#include "potential/src/cpotential.h"
#include "frame/src/cframe.h"

double hamiltonian_value(CPotential *p, CFrameType *fr, double t, double *qp) {
    double v = 0;
    int i;

    v = v + (fr->energy)(t, (fr->parameters), qp, p->n_dim);

    for (i=0; i < p->n_components; i++) {
        // TODO: change potential 'value' -> 'energy'
        v = v + (p->value)[i](t, (p->parameters)[i], qp, p->n_dim);
    }

    return v;
}

void hamiltonian_gradient(CPotential *p, CFrameType *fr, double t, double *qp, double *dH) {
    int i;

    for (i=0; i < 2*(p->n_dim); i++) {
        dH[i] = 0.;
    }

    // potential gradient has to be first
    c_gradient(p, t, qp, &(dH[p->n_dim]));

    (fr->gradient)(t, (fr->parameters), qp, p->n_dim, dH);

    for (i=p->n_dim; i < 2*(p->n_dim); i++) {
        dH[i] = -dH[i]; // pdot = -dH/dq
    }
}

void hamiltonian_hessian(CPotential *p, CFrameType *fr, double t, double *qp, double *d2H) {
    int i;

    for (i=0; i < p->n_components; i++) {
        (p->hessian)[i](t, (p->parameters)[i], qp, p->n_dim, d2H);
    }

    // TODO: not implemented!!
    // TODO: can I just add in the terms from the frame here?
    // (fr->hessian)(t, (fr->parameters), qp, p->n_dim, d2H);
}
