cdef class _CPotential:
    cpdef value(self, double[:,::1] xyz)
    cdef public void _value(self, double[:,::1] xyz, double[::1] pot, int nparticles)

    cpdef gradient(self, double[:,::1] xyz)
    cdef public void _gradient(self, double[:,::1] xyz, double[:,::1] acc, int nparticles)

    cpdef hessian(self, double[:,::1] w)
    cdef public void _hessian(self, double[:,::1] w, double[:,::1] acc, int nparticles)