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

    // TODO: instead of hard-setting 3, I need to define p->ndim
    for (i=0; i < 3; i++) {
        grad[i] = 0.;
    }

    for (i=0; i < p->n_components; i++) {
        (p->gradient)[i](t, (p->parameters)[i], q, grad);
    }
}
