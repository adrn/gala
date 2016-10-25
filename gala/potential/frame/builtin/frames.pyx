# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

# Third-party
import numpy as np
cimport numpy as np
np.import_array()

# Project
from ..cframe import CFrameBase
from ..cframe cimport CFrameWrapper

cdef extern from "src/funcdefs.h":
    ctypedef double (*densityfunc)(double t, double *pars, double *q) nogil
    ctypedef double (*valuefunc)(double t, double *pars, double *q) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, double *hess) nogil

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        valuefunc potential
        gradientfunc gradient
        hessianfunc hessian

        int n_params
        double *parameters;

cdef extern from "frame/builtin/builtin_frames.h":
    double static_frame_potential(double t, double *pars, double *qp) nogil
    void static_frame_gradient(double t, double *pars, double *qp, double *grad) nogil
    void static_frame_hessian(double t, double *pars, double *qp, double *hess) nogil

__all__ = ['StaticFrame']

cdef class StaticFrameWrapper(CFrameWrapper):

    def __init__(self):
        cdef CFrame cf

        cf.potential = <valuefunc>(static_frame_potential)
        cf.gradient = <gradientfunc>(static_frame_gradient)
        cf.hessian = <hessianfunc>(static_frame_hessian)
        cf.n_params = 0
        cf.parameters = NULL # &(self._frame_params[0])

        self.cframe = cf

class StaticFrame(CFrameBase):

    def __init__(self):
        """
        TODO:
        """
        c_instance = StaticFrameWrapper()
        super(StaticFrame, self).__init__(c_instance)

# cdef class ConstantRotatingFrame:
#     pass
