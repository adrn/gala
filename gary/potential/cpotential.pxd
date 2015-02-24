ctypedef double (*valuefunc)(double *pars, double *q) nogil
ctypedef void (*gradientfunc)(double *pars, double *q, double *grad) nogil

cdef class _CPotential:
    cdef double *_parameters
    cdef valuefunc c_value
    cdef gradientfunc c_gradient

    cpdef value(self, double[:,::1] q)
    cdef public double _value(self, double *q) nogil

    cpdef gradient(self, double[:,::1] q)
    cdef public void _gradient(self, double *q, double *grad) nogil

    cpdef hessian(self, double[:,::1] w)
    cdef public void _hessian(self, double *w, double *hess) nogil

    cpdef mass_enclosed(self, double[:,::1] q)
    cdef public double _mass_enclosed(self, double *q, double *epsilon, double Gee) nogil

# cdef public class _CCompositePotential[type _CPotentialType, object _CPotential]:
#     cpdef value(self, double[:,::1] q, double[::1] pot)
#     cdef public double _value(self, double[:,::1] q, int k) nogil

#     cpdef gradient(self, double[:,::1] q)
#     cdef public void _gradient(self, double[:,::1] q, double[:,::1] grad, int k) nogil

#     cpdef hessian(self, double[:,::1] w)
#     cdef public void _hessian(self, double[:,::1] w, double[:,:,::1] hess, int k) nogil

#     cpdef mass_enclosed(self, double[:,::1] q)
#     cdef public double _mass_enclosed(self, double[:,::1] q, double[:,::1] epsilon, double Gee, int k) nogil
