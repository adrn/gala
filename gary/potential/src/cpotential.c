#include "cpotential.h"

double c_value(CPotential *p, double t, double *q) {
    double v = 0;

    for (int i=0; i < p->n_components; i++) {
        v = v + (p->value)[i](t, (p->parameters)[i], q);
        printf("%f %f\n", (p->parameters)[i][0], (p->parameters)[i][1]);
    }
    return v;
}

// double c_density(CPotential p, double t, double *q) {
//     double v = 0;
//     for (int i=0; i < p.n_components; i++) {
//         v = v + p.density[i](t, p.parameters[i], q);
//     }
//     return v;
// }

// void c_gradient(CPotential p, double t, double *q, double *grad) {
//     double v = 0;
//     for (int i=0; i < p.n_components; i++) {
//         v = v + p.gradient[i](t, p.parameters[i], q, grad);
//     }
//     return v;
// }
