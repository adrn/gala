# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

# Standard library
from collections import OrderedDict

# Third-party
from astropy.utils.misc import isiterable
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()

# Project
from ..cframe import CFrameBase
from ..cframe cimport CFrameWrapper
from ....units import dimensionless, DimensionlessUnitSystem

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

    double constant_rotating_frame_hamiltonian_2d(double t, double *pars, double *qp, int n_dim) nogil
    void constant_rotating_frame_gradient_2d(double t, double *pars, double *qp, int n_dim, double *grad) nogil
    void constant_rotating_frame_hessian_2d(double t, double *pars, double *qp, int n_dim, double *hess) nogil

    double constant_rotating_frame_hamiltonian_3d(double t, double *pars, double *qp, int n_dim) nogil
    void constant_rotating_frame_gradient_3d(double t, double *pars, double *qp, int n_dim, double *grad) nogil
    void constant_rotating_frame_hessian_3d(double t, double *pars, double *qp, int n_dim, double *hess) nogil

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
    """
    Represents a static intertial reference frame.

    Parameters
    ----------
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, units=None):
        # ndim = None means it can support any number of ndim
        super(StaticFrame, self).__init__(StaticFrameWrapper, dict(), units, ndim=None)

# ---

cdef class ConstantRotatingFrameWrapper2D(CFrameWrapper):

    def __init__(self, double Omega):
        cdef:
            CFrame cf

        self._params = np.array([Omega], dtype=np.float64)

        cf.energy = <energyfunc>(constant_rotating_frame_hamiltonian_2d)
        cf.gradient = <gradientfunc>(constant_rotating_frame_gradient_2d)
        cf.hessian = <hessianfunc>(constant_rotating_frame_hessian_2d)
        cf.n_params = 1
        cf.parameters = &(self._params[0])

        self.cframe = cf

cdef class ConstantRotatingFrameWrapper3D(CFrameWrapper):

    def __init__(self, double Omega_x, double Omega_y, double Omega_z):
        cdef:
            CFrame cf

        self._params = np.array([Omega_x, Omega_y, Omega_z], dtype=np.float64)

        cf.energy = <energyfunc>(constant_rotating_frame_hamiltonian_3d)
        cf.gradient = <gradientfunc>(constant_rotating_frame_gradient_3d)
        cf.hessian = <hessianfunc>(constant_rotating_frame_hessian_3d)
        cf.n_params = 3
        cf.parameters = &(self._params[0])

        self.cframe = cf

class ConstantRotatingFrame(CFrameBase):
    """
    Represents a constantly rotating reference frame.

    The reference frame rotates with constant angular velocity set by the
    magnitude of the vector parameter ``Omega`` around the axis defined by
    the unit vector computed from the input frequency vector.

    Parameters
    ----------
    Omega : :class:`~astropy.units.Quantity`
        The frequency vector, which specifies the axis of rotation and the
        angular velocity of the frame.
    units : `~gala.units.UnitSystem` (optional)
        Set of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, Omega, units=None):

        if units is None:
            units = dimensionless

        Omega = np.atleast_1d(Omega)

        if Omega.shape == (1,):
            # assumes ndim=2, must be associated with a 2D potential
            Omega = np.squeeze(Omega)
            ndim = 2
            ConstantRotatingFrameWrapper = ConstantRotatingFrameWrapper2D

        elif Omega.shape == (3,):
            # assumes ndim=3, must be associated with a 3D potential
            ndim = 3
            ConstantRotatingFrameWrapper = ConstantRotatingFrameWrapper3D

        else:
            raise ValueError("Invalid input for rotation vector Omega.")

        pars = OrderedDict()
        ptypes = OrderedDict()
        pars['Omega'] = Omega
        ptypes['Omega'] = 'frequency'

        super(ConstantRotatingFrame, self).__init__(ConstantRotatingFrameWrapper,
                                                    pars, units, ndim=ndim)

        if self.parameters['Omega'].unit != u.one and isinstance(units, DimensionlessUnitSystem):
            raise ValueError("If frequency vector has units, you must pass in a unit system.")
