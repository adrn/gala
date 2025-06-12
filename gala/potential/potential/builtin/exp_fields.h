#pragma once

#include <memory>

#include <Coefficients.H>
#include <BiorthBasis.H>

namespace gala_exp {

class State {
public:
    // Keep using shared_ptr for now since Basis is abstract and factory() returns shared_ptr
    // The real fix needs to address the caching issue in BasisClasses::Basis::factory()
    std::shared_ptr<BasisClasses::Basis> basis;
    CoefClasses::CoefsPtr coefs;
    CoefClasses::CoefStrPtr current_coef; // ← CRITICAL: Keep interpolated coef alive!
    bool is_static = false;
};

State exp_init(
    const std::string &config,
    const std::string &coeffile,
    int stride,
    double tmin,
    double tmax,
    int snapshot_index
);

CoefClasses::CoefStrPtr interpolator(double t, CoefClasses::CoefsPtr coefs);

}

extern "C" {

// These must be C function prototypes. Gala will use the C function pointers
// in the evaulation loop.

extern double exp_value(double t, double *pars, double *q, int n_dim, void* state);
extern void exp_gradient(double t, double *pars, double *q, int n_dim, double *grad, void* state);
extern double exp_density(double t, double *pars, double *q, int n_dim, void* state);

}
