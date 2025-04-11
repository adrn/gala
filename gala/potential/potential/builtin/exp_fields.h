#pragma once

#include <memory>

#include <Coefficients.H>
#include <BiorthBasis.H>

namespace gala_exp {

class State {
public:
    std::shared_ptr<BasisClasses::Basis> basis;
    CoefClasses::CoefsPtr coefs;
    bool is_static = false;
};

State exp_init(const char *config, const char *coeffile, int stride, double tmin, double tmax);

}

extern "C" {

// These must be C function prototypes. Gala will use the C function pointers
// in the evaulation loop.

extern double exp_value(double t, double *pars, double *q, int n_dim, void* state);
extern void exp_gradient(double t, double *pars, double *q, int n_dim, double *grad, void* state);
extern double exp_density(double t, double *pars, double *q, int n_dim, void* state);
extern void exp_hessian(double t, double *pars, double *q, int n_dim, double *hess, void* state);

}
