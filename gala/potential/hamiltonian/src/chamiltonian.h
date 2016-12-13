#include "potential/src/cpotential.h"
#include "frame/src/cframe.h"

extern double hamiltonian_value(CPotential *p, CFrame *fr, double t, double *q);
extern void hamiltonian_gradient(CPotential *p, CFrame *fr, double t, double *q, double *grad);
extern void hamiltonian_hessian(CPotential *p, CFrame *fr, double t, double *q, double *hess);
