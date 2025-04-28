# cython: language_level=3

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrameType:
        pass

cdef extern from "potential/src/cpotential.h":
    ctypedef struct CPotential:
        pass

cdef extern from "dopri/dop853.h":
    ctypedef void (*FcnEqDiff)(unsigned n, double x, double *y, double *f,
                              CPotential *p, CFrameType *fr, unsigned norbits,
                              unsigned nbody, void *args) nogil

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

    double contd8(unsigned ii, double x)
    # Thread-safe dense output function
    double contd8_threadsafe(Dop853DenseState *state, unsigned ii, double x)

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
    unsigned save_all=?
)

# cpdef dop853_integrate_hamiltonian(hamiltonian, double[:,::1] w0, double[::1] t,
#                                    double atol=?, double rtol=?, int nmax=?)
