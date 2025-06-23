#include "src/funcdefs.h"

#ifndef _CPotential_H
#define _CPotential_H
    typedef struct _CPotential CPotential;

    struct _CPotential {
        int n_components; // number of potential components
        int n_dim; // coordinate system dimensionality
        int null; // short circuit: if null, can skip evaluation
        int* do_shift_rotate; // short circuit: if 0, skip transforming pos/vel

        // arrays of pointers to each of the function types above
        densityfunc* density;
        energyfunc* value;
        gradientfunc* gradient;
        hessianfunc* hessian;

        // array containing the number of parameters in each component
        int* n_params;

        // pointer to array of pointers to the parameter arrays
        double** parameters;

        // pointer to array of pointers containing the origin coordinates
        double** q0;

        // pointer to array of pointers containing rotation matrix elements
        double** R;

        // pointer to array of pointers containing the state
        void **state;
    };
#endif

extern CPotential* allocate_cpotential(int n_components);
extern void free_cpotential(CPotential* p);
extern int resize_cpotential_arrays(CPotential* pot, int new_n_components);

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
