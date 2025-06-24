#ifndef _FUNCS_
#define _FUNCS_
    typedef double (*densityfunc)(double t, double *pars, double *q, int n_dim, void *state);
    typedef double (*energyfunc)(double t, double *pars, double *q, int n_dim, void *state);
    typedef void (*gradientfunc)(double t, double *pars, double *q, int n_dim, double *grad, void *state);
    typedef void (*hessianfunc)(double t, double *pars, double *q, int n_dim, double *hess, void *state);
#endif
