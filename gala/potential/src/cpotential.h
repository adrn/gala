#ifndef MAX_N_COMPONENTS_H
#define MAX_N_COMPONENTS_H
    enum {MAX_N_COMPONENTS = 16};
#endif

#ifndef _FUNCS_
#define _FUNCS_
    typedef double (*densityfunc)(double t, double *pars, double *q);
    typedef double (*valuefunc)(double t, double *pars, double *q);
    typedef void (*gradientfunc)(double t, double *pars, double *q, double *grad);
#endif

#ifndef _CPotential_H
#define _CPotential_H
    typedef struct _CPotential CPotential;

    struct _CPotential {
        int n_components; // number of potential components
        int n_dim; // coordinate system dimensionality

        // arrays of pointers to each of the function types above
        densityfunc density[MAX_N_COMPONENTS];
        valuefunc value[MAX_N_COMPONENTS];
        gradientfunc gradient[MAX_N_COMPONENTS];

        // array containing the number of parameters in each component
        int n_params[MAX_N_COMPONENTS];

        // pointer to array of pointers to the parameter arrays for each component
        double *parameters[MAX_N_COMPONENTS];
    };
#endif

extern double c_value(CPotential *p, double t, double *q);
extern double c_density(CPotential *p, double t, double *q);
extern void c_gradient(CPotential *p, double t, double *q, double *grad);
extern double c_d_dr(CPotential *p, double t, double *q, double *epsilon);
extern double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon);
extern double c_mass_enclosed(CPotential *p, double t, double *q, double G, double *epsilon);
