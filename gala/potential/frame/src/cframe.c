#include "frame/src/cframe.h"

double frame_hamiltonian(CFrameType *fr, double t, double *qp, int n_dim) {
    double v = (fr->energy)(t, (fr->parameters), qp, n_dim);
    return v;
}

void frame_gradient(CFrameType *fr, double t, double *qp, int n_dim, double *dH) {
    (fr->gradient)(t, (fr->parameters), qp, n_dim, dH);
}

void frame_hessian(CFrameType *fr, double t, double *qp, int n_dim, double *d2H) {
    // TODO: not implemented!!
    // TODO: can I just add in the terms from the frame here?
    // (fr->hessian)(t, (fr->parameters), qp, n_dim, d2H);
}
