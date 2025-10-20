#ifndef VECTORIZATION_H
#define VECTORIZATION_H

#include <stddef.h>
#include <stdexcept>

class double6ptr {
    // This class is a container for 6 pointers to doubles.
    // It lets us naturally pass around arrays of shape (6,N).
    // Note that not all the pointers may be valid; it is the
    // responsibility of the downstream function to not dereference
    // an invalid pointer!

public:
    double *__restrict__ x, *__restrict__ y, *__restrict__ z;
    double *__restrict__ px, *__restrict__ py, *__restrict__ pz;

    explicit double6ptr(double *__restrict__ q, size_t N) {
        x = q;
        y = q + N;
        z = q + 2 * N;
        px = q + 3 * N;
        py = q + 4 * N;
        pz = q + 5 * N;
    }

    // Access through the index operator will dereference the pointers
    double & operator[](int i) {
        switch (i) {
            case 0: return *x;
            case 1: return *y;
            case 2: return *z;
            case 3: return *px;
            case 4: return *py;
            case 5: return *pz;
            default: throw std::out_of_range("Index out of range");
        }
    }

    const double & operator[](int i) const {
        switch (i) {
            case 0: return *x;
            case 1: return *y;
            case 2: return *z;
            case 3: return *px;
            case 4: return *py;
            case 5: return *pz;
            default: throw std::out_of_range("Index out of range");
        }
    }
};

// This is a wrapper to generate vectorized versions of the scalar gradient
// functions. It looks a little gnarly because we have to play some tricks to ensure
// that the compiler can vectorize through the call to the scalar function.

#define DEFINE_VECTORIZED_GRADIENT(POTENTIAL_NAME) \
struct POTENTIAL_NAME##_gradient_functor { \
    template <typename ...Params> \
    void operator()(Params&&... params) { \
        POTENTIAL_NAME##_gradient_single(std::forward<Params>(params)...); \
    } \
}; \
void POTENTIAL_NAME##_gradient(double t, double *__restrict__ pars, double *__restrict__ q, int n_dim, size_t N, double *__restrict__ grad, void *__restrict__ state) { \
    gradientv<POTENTIAL_NAME##_gradient_functor>(t, pars, q, n_dim, N, grad, state); \
}

template <typename F>
void gradientv(double t, double *__restrict__ pars, double *__restrict__ q, int n_dim, size_t N, double *__restrict__ grad, void *__restrict__ state) {
    F f;
    for (size_t i = 0; i < N; i++) {
        f(t, pars,
            double6ptr{q + i, N},
            n_dim,
            double6ptr{grad + i, N},
        state);
    }
}

#endif
