#include "src/funcdefs.h"

#ifndef _CFRAME_H
#define _CFRAME_H
    typedef struct _CFrame CFrame;

    struct _CFrame {
        // arrays of pointers to each of the function types above
        valuefunc potential;
        gradientfunc gradient;
        hessianfunc hessian;

        int n_params;

        // pointer to the parameter array
        double *parameters;
    };
#endif
