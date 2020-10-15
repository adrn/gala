#include "extra_compile_macros.h"

#if USE_GSL == 1
extern double rho_nl(double s, int n, int l);
extern double rho_nlm(double s, double phi, double X, int n, int l, int m);

extern double phi_nl(double s, int n, int l);
extern double phi_nlm(double s, double phi, double X, int n, int l, int m);

extern void sph_grad_phi_nlm(double s, double phi, double X, int n, int l, int m, int lmax, double *sphgrad);
#endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif