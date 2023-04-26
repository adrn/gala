#include "src/funcdefs.h"

#ifndef MAX_N_COMPONENTS_H
    #define MAX_N_COMPONENTS_H
    #define MAX_N_COMPONENTS 16
#endif

#ifndef _CPotential_H
#define _CPotential_H
    typedef struct _CPotential CPotential;

    struct _CPotential {
        int n_components; // number of potential components
        int n_dim; // coordinate system dimensionality
        int null; // a short circuit: if null, can skip evaluation

        // arrays of pointers to each of the function types above
        densityfunc density[MAX_N_COMPONENTS];
        energyfunc value[MAX_N_COMPONENTS];
        gradientfunc gradient[MAX_N_COMPONENTS];
        hessianfunc hessian[MAX_N_COMPONENTS];

        // array containing the number of parameters in each component
        int n_params[MAX_N_COMPONENTS];

        // pointer to array of pointers to the parameter arrays
        double *parameters[MAX_N_COMPONENTS];

        // pointer to array of pointers containing the origin coordinates
        double *q0[MAX_N_COMPONENTS];

        // pointer to array of pointers containing rotation matrix elements
        double *R[MAX_N_COMPONENTS];
    };
#endif

extern double c_potential(CPotential *p, double t, double *q);
extern double c_density(CPotential *p, double t, double *q);
extern void c_gradient(CPotential *p, double t, double *q, double *grad);
extern void c_hessian(CPotential *p, double t, double *q, double *hess);

// TODO: err, what about reference frames...
extern double c_d_dr(CPotential *p, double t, double *q, double *epsilon);
extern double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon);
extern double c_mass_enclosed(CPotential *p, double t, double *q, double G, double *epsilon);

// TODO: move this elsewhere?
void c_nbody_acceleration(CPotential **pots, double t, double *qp,
                          int norbits, int nbody, int ndim, double *acc);
void c_nbody_gradient_symplectic(
    CPotential **pots, double t, double *q,
    double *nbody_q, int nbody, int nbody_i,
    int ndim, double *grad
);