#include "potential/src/cpotential.h"
#include "frame/src/cframe.h"

extern double hamiltonian_value(CPotential *p, CFrameType *fr, double t, double *q);
extern void hamiltonian_gradient(CPotential *p, CFrameType *fr, double t, double *q, double *grad);
extern void hamiltonian_hessian(CPotential *p, CFrameType *fr, double t, double *q, double *hess);
