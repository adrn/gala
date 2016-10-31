#ifndef _FUNCS_
#define _FUNCS_
    typedef double (*densityfunc)(double t, double *pars, double *q);
    typedef double (*energyfunc)(double t, double *pars, double *q);
    typedef void (*gradientfunc)(double t, double *pars, double *q, double *grad);
    typedef void (*hessianfunc)(double t, double *pars, double *q, double *hess);
#endif
