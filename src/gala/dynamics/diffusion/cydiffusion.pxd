# cython: language_level=3
# cython: language=c++

cdef extern from "dynamics/diffusion/src/diffusion.h":
    ctypedef void (*diffusionfunc)(
        double t, double *pars, double *q, double *v, int n_dim,
        double *drift, double *M, void *state
    ) noexcept nogil

    ctypedef struct CDiffusion:
        int n_dim
        int returns_factor
        int n_params
        diffusionfunc func
        double *parameters
        void *state

    void *gala_diffusion_rng_alloc(unsigned long long seed) nogil
    double gala_diffusion_rng_gaussian(void *rng) nogil
    void gala_diffusion_rng_free(void *rng) nogil
    int gala_diffusion_cholesky(const double *M, int n_dim, double *L) nogil

    void constant_diag_diffusion(
        double t, double *pars, double *q, double *v, int n_dim,
        double *drift, double *M, void *state
    ) noexcept nogil
    void constant_tensor_diffusion(
        double t, double *pars, double *q, double *v, int n_dim,
        double *drift, double *M, void *state
    ) noexcept nogil
    void example_radial_diffusion(
        double t, double *pars, double *q, double *v, int n_dim,
        double *drift, double *M, void *state
    ) noexcept nogil


cdef class CDiffusionWrapper:
    cdef CDiffusion diffusion
    cdef double[::1] _params
    cpdef init(self, double[::1] parameters, int n_dim, int returns_factor)
    cdef CDiffusion* get_ptr(self) noexcept nogil
