cdef class _CPotential:
    cpdef value(self, double[:,::1] xyz)
    cdef public void _value(self, double[:,::1] xyz, double[::1] pot, int nparticles) nogil

    cpdef gradient(self, double[:,::1] xyz)
    cdef public void _gradient(self, double[:,::1] xyz, double[:,::1] grad, int nparticles) nogil

    cpdef hessian(self, double[:,::1] w)
    cdef public void _hessian(self, double[:,::1] w, double[:,::1] hess, int nparticles) nogil

    cpdef acceleration(self, double[:,::1] xyz)
    cdef public void _acceleration(self, double[:,::1] xyz, double[:,::1] acc, int nparticles) nogil

    cpdef tidal_radius(self, double m, double[:,::1] xyz)
    cdef public double _tidal_radius(self, double m, double x, double y, double z) nogil
