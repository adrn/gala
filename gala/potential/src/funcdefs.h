#ifndef _FUNCS_
#define _FUNCS_
    typedef double (*densityfunc)(double t, double *pars, double *q, int n_dim);
    typedef double (*energyfunc)(double t, double *pars, double *q, int n_dim);
    typedef void (*gradientfunc)(double t, double *pars, double *q, int n_dim, double *grad);
    typedef void (*hessianfunc)(double t, double *pars, double *q, int n_dim, double *hess);
#endif


