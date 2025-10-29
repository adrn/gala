#pragma once

#include <memory>
#include <filesystem>
#include <stdexcept>

#include <EXP/Coefficients.H>
#include <EXP/BiorthBasis.H>

namespace gala_exp {

class State {
public:
    BasisClasses::BasisPtr basis;
    CoefClasses::CoefsPtr coefs;
    double tmin;
    double tmax;
    bool is_static;
    double snapshot_time_factor;
};

State exp_init(
    const std::string &config,
    const std::string &coeffile,
    int stride,
    double tmin,
    double tmax,
    int snapshot_index,
    double snapshot_time_factor
);

CoefClasses::CoefStrPtr interpolator(double t, CoefClasses::CoefsPtr coefs);

}

extern double exp_value(double t, double *pars, double *q, int n_dim, void* state);
extern void exp_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state);
extern double exp_density(double t, double *pars, double *q, int n_dim, void* state);

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
