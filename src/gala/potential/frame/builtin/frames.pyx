# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3
# cython: language=c++

import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()


from ..cframe import CFrameBase
from ..cframe cimport CFrameWrapper, CFrameType
from ...common import PotentialParameter
from ....units import dimensionless, DimensionlessUnitSystem
from ...potential.cpotential cimport energyfunc, gradientfunc, hessianfunc


cdef extern from "frame/builtin/builtin_frames.h":
    double static_frame_hamiltonian(double t, double *pars, double *qp, int n_dim) nogil
    void static_frame_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state) nogil
    void static_frame_hessian(double t, double *pars, double *qp, int n_dim, double *hess) nogil

    double constant_rotating_frame_2d_hamiltonian(double t, double *pars, double *qp, int n_dim) nogil
    void constant_rotating_frame_2d_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state) nogil
    void constant_rotating_frame_2d_hessian(double t, double *pars, double *qp, int n_dim, double *hess) nogil

    double constant_rotating_frame_3d_hamiltonian(double t, double *pars, double *qp, int n_dim) nogil
    void constant_rotating_frame_3d_gradient(double t, double *pars, double *q, int n_dim, size_t N, double *grad, void *state) nogil
    void constant_rotating_frame_3d_hessian(double t, double *pars, double *qp, int n_dim, double *hess) nogil

__all__ = ['StaticFrame', 'ConstantRotatingFrame']

cdef class StaticFrameWrapper(CFrameWrapper):

    def __init__(self, list params):
        cdef CFrameType cf

        self.init(params)

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
    Wrapper = StaticFrameWrapper
    ndim = None

    # NOTE: this is a workaround to allow units as a positional arg for this
    # class, because a lot of existing code assumes that...
    def __init__(self, units=None):
        super().__init__(units=units)

# ---

cdef class ConstantRotatingFrameWrapper2D(CFrameWrapper):

    def __init__(self, list params):
        cdef:
            CFrameType cf

        assert len(params) == 1
        self._params = np.array([params[0]], dtype=np.float64)

        cf.energy = <energyfunc>(constant_rotating_frame_2d_hamiltonian)
        cf.gradient = <gradientfunc>(constant_rotating_frame_2d_gradient)
        cf.hessian = <hessianfunc>(constant_rotating_frame_2d_hessian)
        cf.n_params = 1
        cf.parameters = &(self._params[0])

        self.cframe = cf

cdef class ConstantRotatingFrameWrapper3D(CFrameWrapper):

    def __init__(self, list params):
        cdef:
            CFrameType cf

        assert len(params) == 3
        self._params = np.array([params[0], params[1], params[2]],
                                dtype=np.float64)

        cf.energy = <energyfunc>(constant_rotating_frame_3d_hamiltonian)
        cf.gradient = <gradientfunc>(constant_rotating_frame_3d_gradient)
        cf.hessian = <hessianfunc>(constant_rotating_frame_3d_hessian)
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
    Omega = PotentialParameter(
        'Omega',
        physical_type='frequency',
        equivalencies=u.dimensionless_angles(),
        ndim=1,
        convert=np.atleast_1d
    )

    def _setup_frame(self, parameters, parameter_is_default, units=None):
        super()._setup_frame(parameters, parameter_is_default, units=units)

        Omega = np.atleast_1d(self.parameters['Omega'])

        if Omega.shape == (1,):
            # assumes ndim=2, must be associated with a 2D potential
            self.ndim = 2
            self.Wrapper = ConstantRotatingFrameWrapper2D

        elif Omega.shape == (3,):
            # assumes ndim=3, must be associated with a 3D potential
            self.ndim = 3
            self.Wrapper = ConstantRotatingFrameWrapper3D

        else:
            raise ValueError("Invalid input for rotation vector Omega.")
