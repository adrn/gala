#include "src/funcdefs.h"

#ifndef _CFRAME_H
#define _CFRAME_H
    typedef struct _CFrame CFrame;

    struct _CFrame {
        // arrays of pointers to each of the function types above
        energyfunc energy;
        gradientfunc gradient;
        hessianfunc hessian;

        int n_params;

        // pointer to the parameter array
        double *parameters;
    };
#endif

extern double frame_hamiltonian(CFrame *fr, double t, double *qp, int n_dim);
extern void frame_gradient(CFrame *fr, double t, double *qp, int n_dim, double *dH);
extern void frame_hessian(CFrame *fr, double t, double *qp, int n_dim, double *d2H);
