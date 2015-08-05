ctypedef double (*valuefunc)(double t, double *pars, double *q) nogil
ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil

cdef class _CPotential:
    cdef double *_parameters
    cdef valuefunc c_value
    cdef gradientfunc c_gradient
    cdef double[::1] _parvec # need to maintain a reference to parameter array

    cpdef value(self, double[:,::1] q)
    cdef public double _value(self, double *q) nogil

    cpdef gradient(self, double[:,::1] q)
    cdef public void _gradient(self, double *q, double *grad) nogil

    cpdef hessian(self, double[:,::1] w)
    cdef public void _hessian(self, double *w, double *hess) nogil

    cpdef mass_enclosed(self, double[:,::1] q, double G)
    cdef public double _mass_enclosed(self, double *q, double *epsilon, double Gee) nogil

# cdef class _CCompositePotential: #[type _CPotentialType, object _CPotential]:

#     cdef public int n  # number of potential components
#     cdef public double G  # gravitational constant in proper units
#     cdef public _CPotential[::1] cpotentials
#     cdef int[::1] pointers # points to array of pointers to C instances
#     cdef int[::1] param_pointers # points to array of pointers to C instances
#     cdef int * _pointers # points to array of pointers to C instances
#     cdef int * _param_pointers # points to array of pointers to C instances

#     cpdef value(self, double[:,::1] q)
#     cdef public double _value(self, double *q) nogil

#     # cpdef gradient(self, double[:,::1] q)
#     # cdef public void _gradient(self, double *q, double *grad) nogil

#     # cpdef hessian(self, double[:,::1] w)
#     # cdef public void _hessian(self, double *w, double *hess) nogil

#     # cpdef mass_enclosed(self, double[:,::1] q, double G)
#     # cdef public double _mass_enclosed(self, double *q, double *epsilon, double Gee) nogil
