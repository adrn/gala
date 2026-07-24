# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3
# cython: language=c++

"""
Cython wrapper holding the C diffusion model (``CDiffusion``) and a ``DiffusionKick``
hand-off struct. A PyCapsule of the kick struct is passed to the leapfrog integrator,
which calls the kick through the function pointer (so no GSL is linked into leapfrog).
"""

import numpy as np
cimport numpy as np
np.import_array()

from cpython.pycapsule cimport PyCapsule_New


cdef extern from "dynamics/diffusion/src/diffusion_kick.h":
    ctypedef void (*kickfunc)(double t, double dt, double *x, double *v,
                              double *dx, double *dv, void *state, void *rng) noexcept nogil
    ctypedef struct DiffusionKick:
        kickfunc kick
        void *state
        void *rng


cdef extern from "dynamics/diffusion/src/diffusion.h":
    ctypedef void (*diffusionfunc)(double t, double *pars, double *x, double *v,
                                   double *mu, double *M, void *state) noexcept nogil

    ctypedef struct CDiffusion:
        int basis
        int returns_factor
        int n_params
        diffusionfunc func
        double *parameters
        void *state

    void* gala_diffusion_rng_alloc(unsigned long long seed) nogil
    void gala_diffusion_rng_set(void* rng, unsigned long long seed) nogil
    void gala_diffusion_rng_free(void* rng) nogil
    void diffusion_kick_increments(double t, double dt, double *x, double *v,
                                   double *dx, double *dv, void *state, void *rng) noexcept nogil
    void constant_diffusion(double t, double *pars, double *x, double *v,
                            double *mu, double *M, void *state) noexcept nogil
    void gridded_cyl_diffusion(double t, double *pars, double *x, double *v,
                               double *mu, double *M, void *state) noexcept nogil
    void* gala_diffusion_grid_alloc(const double *R, int nR, const double *z, int nz,
                                    const double *fields) nogil
    void gala_diffusion_grid_free(void* grid) nogil


# Basis identifiers (must match diffusion.h)
BASIS_CARTESIAN = 0
BASIS_CYLINDRICAL = 1

# Coefficient field count: 6 drift + 21 upper-triangular tensor (must match header)
NFIELDS = 27


cdef class CDiffusionWrapper:
    cdef CDiffusion cdiffusion
    cdef DiffusionKick kick_struct
    cdef double[::1] _params
    cdef void* _rng
    cdef void* _grid
    cdef object _keepalive

    def __cinit__(self):
        self._rng = NULL
        self._grid = NULL
        self._keepalive = None
        self._rng = gala_diffusion_rng_alloc(0)
        if self._rng == NULL:
            raise RuntimeError(
                "Could not allocate a random number generator. The diffusion "
                "integrator requires gala to be built with GSL support."
            )

    def __dealloc__(self):
        if self._grid != NULL:
            gala_diffusion_grid_free(self._grid)
            self._grid = NULL
        if self._rng != NULL:
            gala_diffusion_rng_free(self._rng)
            self._rng = NULL

    cdef void _finish(self) noexcept:
        self.kick_struct.kick = <kickfunc>diffusion_kick_increments
        self.kick_struct.state = <void*>&self.cdiffusion
        self.kick_struct.rng = self._rng

    def init_constant(self, double[::1] params, int basis):
        self._params = params
        self.cdiffusion.basis = basis
        self.cdiffusion.returns_factor = 0
        self.cdiffusion.n_params = params.shape[0]
        self.cdiffusion.parameters = &params[0]
        self.cdiffusion.func = <diffusionfunc>constant_diffusion
        self.cdiffusion.state = NULL
        self._finish()

    def init_gridded(self, double[::1] R, double[::1] z, double[::1] fields_flat,
                     int nR, int nz, int basis):
        self._keepalive = (np.asarray(R), np.asarray(z), np.asarray(fields_flat))
        self._grid = gala_diffusion_grid_alloc(&R[0], nR, &z[0], nz, &fields_flat[0])
        if self._grid == NULL:
            raise RuntimeError(
                "Failed to build the diffusion coefficient grid. The gridded "
                "diffusion model requires gala to be built with GSL support."
            )
        self.cdiffusion.basis = basis
        self.cdiffusion.returns_factor = 0
        self.cdiffusion.n_params = 0
        self.cdiffusion.parameters = NULL
        self.cdiffusion.func = <diffusionfunc>gridded_cyl_diffusion
        self.cdiffusion.state = self._grid
        self._finish()

    def set_seed(self, seed):
        gala_diffusion_rng_set(self._rng, <unsigned long long>seed)

    def kick_capsule(self):
        return PyCapsule_New(<void*>&self.kick_struct, "gala.DiffusionKick", NULL)
