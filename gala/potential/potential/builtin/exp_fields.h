#pragma once

#include <memory>
#include <filesystem>
#include <stdexcept>

#include <Coefficients.H>
#include <BiorthBasis.H>

namespace gala_exp {

class State {
public:
    BasisClasses::BasisPtr basis;
    CoefClasses::CoefsPtr coefs;
    double tmin;
    double tmax;
    bool is_static;
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

class ScopedChdir {
private:
    std::filesystem::path original_path;
    bool empty;

public:
    inline explicit ScopedChdir(const std::filesystem::path& new_path) {
        empty = new_path.empty();

        if(!empty){
            original_path = std::filesystem::current_path();
            std::filesystem::current_path(new_path);
        }
    }

    inline ~ScopedChdir() {
        if (empty) return;
        try {
            std::filesystem::current_path(original_path);
        } catch (...) {
            // Can't throw in destructor
        }
    }

    ScopedChdir(const ScopedChdir&) = delete;
    ScopedChdir& operator=(const ScopedChdir&) = delete;
};
