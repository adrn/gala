# cython: language_level=3
# cython: language=c++

cdef extern from "src/funcdefs.h":
    ctypedef double (*densityfunc)(double t, double *pars, double *q, int n_dim, void *state) except + nogil
    ctypedef double (*energyfunc)(double t, double *pars, double *q, int n_dim, void *state) except + nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state) except + nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, int n_dim, double *hess, void *state) except + nogil

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        int n_components      # number of potential components
        int n_dim             # coordinate system dimensionality
        int null              # shortcut flag to skip evaluation
        int* do_shift_rotate   # shortcut flag to skip pos/vel transformation
        densityfunc* density
        energyfunc* value
        gradientfunc* gradient
        hessianfunc* hessian

        int* n_params         # parameter counts per component
        double** parameters   # pointers to parameter arrays per component
        double** q0           # pointers to origin per component
        double** R            # pointers to rotation per component
        void **state          # pointers to additional state/metadata information

    CPotential* allocate_cpotential(int n_components)
    void free_cpotential(CPotential* p) nogil
    int resize_cpotential_arrays(CPotential* p, int n_components) nogil

    double c_potential(CPotential *p, double t, double *q) except + nogil
    double c_density(CPotential *p, double t, double *q) except + nogil
    void c_gradient(CPotential *p, size_t N, double t, double *q, double *grad) except + nogil
    void c_hessian(CPotential *p, double t, double *q, double *hess) except + nogil

    double c_d_dr(CPotential *p, double t, double *q, double *epsilon) except + nogil
    double c_d2_dr2(CPotential *p, double t, double *q, double *epsilon) except + nogil
    double c_mass_enclosed(CPotential *p, double t, double *q, double G, double *epsilon) except + nogil

    void c_nbody_gradient_symplectic(
        CPotential **pots, double t, double *q,
        double *nbody_q, int nbody, int nbody_i,
        int ndim, double *grad
    ) except + nogil

    void c_nbody_acceleration(CPotential **pots, double t, double *qp,
        int norbits, int nbody, int ndim, double *acc) except + nogil

cpdef _validate_pos_arr(double[:,::1] arr)

cdef class CPotentialWrapper:
    cdef CPotential* cpotential
    cdef double[::1] _params
    cdef int[::1] _n_params
    cdef list _potentials # HACK: for CCompositePotentialWrapper
    cdef double[::1] _q0
    cdef double[::1] _R

    cpdef init(self, list parameters, double[::1] q0, double[:, ::1] R,
               int n_dim=?)

    cpdef energy(self, double[:,::1] q, double[::1] t)
    cpdef density(self, double[:,::1] q, double[::1] t)
    cpdef gradient(self, double[:,::1] q, double[::1] t)
    cpdef hessian(self, double[:,::1] q, double[::1] t)

    cpdef d_dr(self, double[:,::1] q, double G, double[::1] t)
    cpdef d2_dr2(self, double[:,::1] q, double G, double[::1] t)
    cpdef mass_enclosed(self, double[:,::1] q, double G, double[::1] t)
