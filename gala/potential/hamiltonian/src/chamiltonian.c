#include <math.h>

// TODO: Frame has to have parameters too

double hamiltonian_value(CPotential *p, CFrame *fr, double t, double *qp) {
    double v = 0;
    int i;

    for (i=0; i < p->n_components; i++) {
        v = v + (p->value)[i](t, (p->parameters)[i], qp);
    }

    v = v + (fr->potential)(t, (fr->parameters), qp);

    return v;
}

void hamiltonian_gradient(CPotential *p, CFrame *fr, double t, double *qp, double *dH) {
    int i;

    for (i=0; i < 2*(p->n_dim); i++) {
        dH[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        (p->gradient)[i](t, (p->parameters)[i], qp, dH);
    }

    (fr->gradient)(t, (fr->parameters), qp, dH);

}

void hamiltonian_hessian(CPotential *p, CFrame *fr, double t, double *qp, double *d2H) {
    int i;

    for (i=0; i < pow(2*(p->n_dim),2); i++) {
        d2H[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        (p->hessian)[i](t, (p->parameters)[i], qp, d2H);
    }

    // TODO: can I just add in the terms from the frame here?
    // (fr->hessian)(t, (fr->parameters), qp, d2H);
}
