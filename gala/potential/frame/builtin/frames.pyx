# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()

# Project
from ..cframe import CFrameBase
from ..cframe cimport CFrameWrapper
from ....units import dimensionless

cdef extern from "src/funcdefs.h":
    ctypedef double (*energyfunc)(double t, double *pars, double *q, int n_dim) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, int n_dim, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, int n_dim, double *hess) nogil

cdef extern from "frame/src/cframe.h":
    ctypedef struct CFrame:
        energyfunc energy
        gradientfunc gradient
        hessianfunc hessian

        int n_params
        double *parameters;

cdef extern from "frame/builtin/builtin_frames.h":
    double static_frame_hamiltonian(double t, double *pars, double *qp, int n_dim) nogil
    void static_frame_gradient(double t, double *pars, double *qp, int n_dim, double *grad) nogil
    void static_frame_hessian(double t, double *pars, double *qp, int n_dim, double *hess) nogil

    double constant_rotating_frame_hamiltonian(double t, double *pars, double *qp, int n_dim) nogil
    void constant_rotating_frame_gradient(double t, double *pars, double *qp, int n_dim, double *grad) nogil
    void constant_rotating_frame_hessian(double t, double *pars, double *qp, int n_dim, double *hess) nogil

__all__ = ['StaticFrame', 'ConstantRotatingFrame']

cdef class StaticFrameWrapper(CFrameWrapper):

    def __init__(self):
        cdef CFrame cf

        cf.energy = <energyfunc>(static_frame_hamiltonian)
        cf.gradient = <gradientfunc>(static_frame_gradient)
        cf.hessian = <hessianfunc>(static_frame_hessian)
        cf.n_params = 0
        cf.parameters = NULL

        self.cframe = cf

class StaticFrame(CFrameBase):

    def __init__(self, units=None):
        """
        TODO:
        """
        super(StaticFrame, self).__init__(StaticFrameWrapper, dict(), units)

# ---

cdef class ConstantRotatingFrameWrapper(CFrameWrapper):

    def __init__(self, double Omega_x, double Omega_y, double Omega_z):
        cdef:
            CFrame cf

        self._params = np.array([Omega_x, Omega_y, Omega_z], dtype=np.float64)

        cf.energy = <energyfunc>(constant_rotating_frame_hamiltonian)
        cf.gradient = <gradientfunc>(constant_rotating_frame_gradient)
        cf.hessian = <hessianfunc>(constant_rotating_frame_hessian)
        cf.n_params = 3
        cf.parameters = &(self._params[0])

        self.cframe = cf

class ConstantRotatingFrame(CFrameBase):

    def __init__(self, Omega, units=None):
        """
        TODO: write docstring
        TODO: always convert to rad/<time unit> ?
        """
        self.Omega = Omega
        if units is None and not hasattr(Omega, 'unit'):
            units = dimensionless
            self.Omega = self.Omega*u.one

        if not hasattr(self.Omega, 'unit'):
            raise TypeError('Input rotation vector must be a Quantity.')

        self.Omega = self.Omega.to(units['angle']/units['time'], u.dimensionless_angles())

        if not self.Omega.shape == (3,):
            raise ValueError("Invalid input for rotation vector Omega.")

        p = OrderedDict()
        p['Omega_x'] = self.Omega[0].value
        p['Omega_y'] = self.Omega[1].value
        p['Omega_z'] = self.Omega[2].value
        super(ConstantRotatingFrame, self).__init__(ConstantRotatingFrameWrapper,
                                                    p, units)
