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
    ctypedef double (*valuefunc)(double t, double *pars, double *q) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, double *hess) nogil

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        valuefunc energy
        gradientfunc gradient
        hessianfunc hessian

        int n_params
        double *parameters;

cdef extern from "frame/builtin/builtin_frames.h":
    double static_frame_hamiltonian(double t, double *pars, double *qp) nogil
    void static_frame_gradient(double t, double *pars, double *qp, double *grad) nogil
    void static_frame_hessian(double t, double *pars, double *qp, double *hess) nogil

    double constant_rotating_frame_hamiltonian(double t, double *pars, double *qp) nogil
    void constant_rotating_frame_gradient(double t, double *pars, double *qp, double *grad) nogil
    void constant_rotating_frame_hessian(double t, double *pars, double *qp, double *hess) nogil

__all__ = ['StaticFrame'] #, 'ConstantRotatingFrame']

cdef class StaticFrameWrapper(CFrameWrapper):

    def __init__(self):
        cdef CFrame cf

        cf.energy = <valuefunc>(static_frame_hamiltonian)
        cf.gradient = <gradientfunc>(static_frame_gradient)
        cf.hessian = <hessianfunc>(static_frame_hessian)
        cf.n_params = 0
        cf.parameters = NULL

        self.cframe = cf

class StaticFrame(CFrameBase):

    def __init__(self):
        """
        TODO:
        """
        c_instance = StaticFrameWrapper()
        super(StaticFrame, self).__init__(c_instance)

# ---

cdef class ConstantRotatingFrameWrapper(CFrameWrapper):

    def __init__(self, double[::1] Omega):
        cdef CFrame cf

        cf.energy = <valuefunc>(constant_rotating_frame_hamiltonian)
        cf.gradient = <gradientfunc>(constant_rotating_frame_gradient)
        cf.hessian = <hessianfunc>(constant_rotating_frame_hessian)
        cf.n_params = 3
        cf.parameters = &(Omega[0])

        self.cframe = cf

class ConstantRotatingFrame(CFrameBase):

    def __init__(self, Omega):
        """
        TODO: write docstring
        TODO: check Omega
        TODO: fuck, this does need to know about units to convert parameters...
        """
        Omega = np.array(Omega)
        c_instance = ConstantRotatingFrameWrapper(Omega)
        super(ConstantRotatingFrame, self).__init__(c_instance)
