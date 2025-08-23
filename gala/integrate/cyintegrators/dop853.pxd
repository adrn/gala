# cython: language_level=3
# cython: language=c++

from libc.stdio cimport FILE

from ...potential.potential.cpotential cimport CPotential
from ...potential.frame.cframe cimport CFrameType

cdef extern from "dopri/dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f,
                              CPotential *p, CFrameType *fr, unsigned norbits,
                              unsigned nbody, void *args) nogil
    ctypedef void (*SolTrait)(long nr, double xold, double x, double* y,
                              unsigned n, int* irtrn)
    ctypedef struct Dop853DenseState:
        double *rcont1
        double *rcont2
        double *rcont3
        double *rcont4
        double *rcont5
        double *rcont6
        double *rcont7
        double *rcont8
        double xold
        double hout
        unsigned nrds
        unsigned *indir

    void Fwrapper (unsigned ndim, double t, double *w, double *f,
                   CPotential *p, CFrameType *fr, unsigned norbits, unsigned nbody) except +
    void Fwrapper_T (unsigned ndim, double t, double *w, double *f,
                   CPotential *p, CFrameType *fr, unsigned norbits, unsigned nbody) except +
    void Fwrapper_direct_nbody(unsigned ndim, double t, double *w, double *f,
                               CPotential *p, CFrameType *fr,
                               unsigned norbits, unsigned nbody, void *args) except + nogil

    # See dop853.h for full description of all input parameters
    int dop853 (unsigned n, FcnEqDiff fn,
                CPotential *p, CFrameType *fr, unsigned n_orbits, unsigned nbody,
                void *args,
                double x, double* y, double xend,
                double* rtoler, double* atoler, int itoler, SolTrait solout,
                int iout, FILE* fileout, double uround, double safe, double fac1,
                double fac2, double beta, double hmax, double h, long nmax, int meth,
                long nstiff, unsigned nrdens, unsigned* icont, unsigned licont,
                Dop853DenseState* dense_state,
                double* tout, unsigned ntout, double* yout) except +

    Dop853DenseState* dop853_dense_state_alloc(unsigned nrdens, unsigned n) nogil
    void dop853_dense_state_free(Dop853DenseState* state, unsigned n) nogil

    double contd8(unsigned ii, double x)
    # Thread-safe dense output function
    double contd8_threadsafe(Dop853DenseState *state, unsigned ii, double x)

    double six_norm (double *x)

cdef void dop853_step(CPotential *cp, CFrameType *cf, FcnEqDiff F,
                      double *w, double t1, double t2, double dt0,
                      int ndim, int norbits, int nbody, void *args,
                      double atol, double rtol, int nmax,
                      unsigned err_if_fail, unsigned log_output,)

cdef dop853_helper(
    CPotential *cp, CFrameType *cf, FcnEqDiff F,
    double[:,::1] w0, double[::1] t,
    int ndim, int norbits, int nbody, void *args, int ntimes,
    double atol, double rtol, int nmax, double dt_max,
    int nstiff,
    unsigned err_if_fail,
    unsigned log_output,
    int transposed,
    unsigned save_all=?
)

# cpdef dop853_integrate_hamiltonian(hamiltonian, double[:,::1] w0, double[::1] t,
#                                    double atol=?, double rtol=?, int nmax=?)
