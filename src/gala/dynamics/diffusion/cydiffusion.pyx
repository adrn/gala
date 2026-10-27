# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3
# cython: language=c++

"""
Cython wrapper holding the C diffusion model (``CDiffusion``). Each integration gets
its own ``DiffusionKick`` (with its own RNG) wrapped in a PyCapsule that keeps the
model alive; the leapfrog integrator calls the kick through the function pointer (so no
GSL is linked into leapfrog).
"""

import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject, Py_INCREF, Py_XDECREF
from cpython.pycapsule cimport (
    PyCapsule_New, PyCapsule_GetPointer, PyCapsule_SetContext, PyCapsule_GetContext
)

from .diffusion_kick cimport DiffusionKick, kickfunc


cdef extern from "dynamics/diffusion/src/diffusion.h":
    ctypedef void (*diffusionfunc)(double t, double *pars, double *x, double *v,
                                   double *mu, double *M, void *state) noexcept nogil

    ctypedef struct CDiffusion:
        int basis
        int returns_factor
        diffusionfunc func
        double *parameters
        void *state

    void* gala_diffusion_rng_alloc(unsigned long long seed) nogil
    void gala_diffusion_rng_free(void* rng) nogil
    int gala_diffusion_cholesky(const double *M, int n, double *L) nogil
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

cdef const char* _CAPSULE_NAME = "gala.DiffusionKick"


cdef void _free_kick_capsule(object capsule) noexcept:
    cdef DiffusionKick* dk = <DiffusionKick*>PyCapsule_GetPointer(capsule, _CAPSULE_NAME)
    gala_diffusion_rng_free(dk.rng)
    free(dk)
    Py_XDECREF(<PyObject*>PyCapsule_GetContext(capsule))


cdef class CDiffusionWrapper:
    cdef CDiffusion cdiffusion
    cdef double[::1] _params
    cdef void* _grid

    def __cinit__(self):
        self._grid = NULL

    def __dealloc__(self):
        if self._grid != NULL:
            gala_diffusion_grid_free(self._grid)
            self._grid = NULL

    def init_constant(self, double[::1] params, int basis):
        """params = [mu(6), D(36)]; D is replaced by its Cholesky factor once here."""
        cdef double[::1] p = np.array(params, copy=True)
        gala_diffusion_cholesky(&params[6], 6, &p[6])
        self._params = p
        self.cdiffusion.basis = basis
        self.cdiffusion.returns_factor = 1
        self.cdiffusion.parameters = &p[0]
        self.cdiffusion.func = <diffusionfunc>constant_diffusion
        self.cdiffusion.state = NULL

    def init_gridded(self, double[::1] R, double[::1] z, double[::1] fields_flat,
                     int nR, int nz, int basis):
        self._grid = gala_diffusion_grid_alloc(&R[0], nR, &z[0], nz, &fields_flat[0])
        if self._grid == NULL:
            raise RuntimeError(
                "Failed to build the diffusion coefficient grid. The gridded "
                "diffusion model requires gala to be built with GSL support."
            )
        self.cdiffusion.basis = basis
        self.cdiffusion.returns_factor = 0
        self.cdiffusion.parameters = NULL
        self.cdiffusion.func = <diffusionfunc>gridded_cyl_diffusion
        self.cdiffusion.state = self._grid

    def kick_capsule(self, unsigned long long seed):
        """A new kick (own RNG, seeded with ``seed``) that keeps this model alive."""
        cdef void* rng = gala_diffusion_rng_alloc(seed)
        if rng == NULL:
            raise RuntimeError(
                "Could not allocate a random number generator. The diffusion "
                "integrator requires gala to be built with GSL support."
            )
        cdef DiffusionKick* dk = <DiffusionKick*>malloc(sizeof(DiffusionKick))
        if dk == NULL:
            gala_diffusion_rng_free(rng)
            raise MemoryError()
        dk.kick = <kickfunc>diffusion_kick_increments
        dk.state = <void*>&self.cdiffusion
        dk.rng = rng
        capsule = PyCapsule_New(<void*>dk, _CAPSULE_NAME, _free_kick_capsule)
        Py_INCREF(self)
        PyCapsule_SetContext(capsule, <void*>self)
        return capsule
