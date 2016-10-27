#include "frame/src/cframe.h"

double frame_hamiltonian(CFrame *fr, double t, double *qp) {
    double v = (fr->energy)(t, (fr->parameters), qp);
    return v;
}

void frame_gradient(CFrame *fr, double t, double *qp, double *dH) {
    (fr->gradient)(t, (fr->parameters), qp, dH);
}

void frame_hessian(CFrame *fr, double t, double *qp, double *d2H) {
    // TODO: not implemented!!
    // TODO: can I just add in the terms from the frame here?
    // (fr->hessian)(t, (fr->parameters), qp, d2H);
}
