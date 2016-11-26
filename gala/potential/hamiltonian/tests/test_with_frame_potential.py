# coding: utf-8

# Third-party
import astropy.units as u
import pytest

# Project
from .. import Hamiltonian
from ...potential.builtin import SphericalNFWPotential, KeplerPotential
from ...frame.builtin import StaticFrame, ConstantRotatingFrame
from ...tests.helpers import _TestBase
from ....units import galactic, dimensionless
from ....dynamics import CartesianPhaseSpacePosition
from ....integrate import DOPRI853Integrator

# ----------------------------------------------------------------------------

import astropy.units as u
import numpy as np
from gala.dynamics import PhaseSpacePosition, Orbit

def to_rotating_frame(omega, w, t=None):
    """
    TODO: figure out units shit for omega and t
    """

    if not hasattr(omega, 'unit'):
        raise TypeError("Input frequency vector must be a Quantity object.")

    try:
        omega = omega.to(u.rad/u.Myr, equivalencies=u.dimensionless_angles()).value
    except:
        omega = omega.value

    if isinstance(w, Orbit) and t is not None:
        raise TypeError("If passing in an Orbit object, do not also specify "
                        "a time array, t.")

    elif not isinstance(w, Orbit) and t is None:
        raise TypeError("If not passing in an Orbit object, you must also specify "
                        "a time array, t.")

    elif t is not None and not hasattr(t, 'unit'):
        raise TypeError("Input time must be a Quantity object.")

    if t is not None:
        t = np.atleast_1d(t) # works with Quantity's
    else:
        t = w.t

    try:
        t = t.to(u.Myr).value
    except:
        t = t.value

    if isinstance(w, PhaseSpacePosition) or isinstance(w, Orbit):
        Cls = w.__class__
        x_shape = w.pos.shape
        x_unit = w.pos.unit
        v_unit = w.vel.unit

        x = w.pos.reshape(3,-1).value
        v = w.vel.reshape(3,-1).value

    else:
        Cls = None
        ndim = w.shape[0]
        x_shape = (ndim//2,) + w.shape[1:]
        x = w[:ndim//2]
        v = w[ndim//2:]

        if hasattr(x, 'unit'):
            raise TypeError("If w is not an Orbit or PhaseSpacePosition, w "
                            "cannot have units!")

        x_unit = u.one
        v_unit = u.one

    # now need to compute rotation vector, ee, and angle, theta
    ee = omega / np.linalg.norm(omega)
    theta = (np.linalg.norm(omega) * t)[None]

    # we use Rodrigues' rotation formula to rotate the position
    x_rot = np.cos(theta)*x + np.sin(theta)*np.cross(ee, x, axisa=0, axisb=0, axisc=0) \
        + (1 - np.cos(theta)) * np.einsum("i,ij->j", ee, x) * ee[:,None]

    v_cor = np.cross(omega, x, axisa=0, axisb=0, axisc=0) * x_unit
    v_rot = v - v_cor.to(v_unit, u.dimensionless_angles()).value

    x_rot = x_rot.reshape(x_shape) * x_unit
    v_rot = v_rot.reshape(x_shape) * v_unit

    if Cls is None:
        return np.vstack((x_rot, v_rot))

    else:
        if issubclass(Cls, Orbit):
            return Cls(pos=x_rot, vel=v_rot, t=t)
        else:
            return Cls(pos=x_rot, vel=v_rot)

# ----------------------------------------------------------------------------

class TestLogPotentialStaticFrame(_TestBase):
    obj = Hamiltonian(SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic),
                      StaticFrame(units=galactic))

    @pytest.mark.skip("Not implemented")
    def test_hessian(self):
        pass

class TestKeplerRotatingFrame(_TestBase):
    obj = Hamiltonian(KeplerPotential(m=1., units=dimensionless),
                      ConstantRotatingFrame(Omega=[0,0,1.], units=dimensionless))

    @pytest.mark.skip("Not implemented")
    def test_hessian(self):
        pass

    def test_integrate(self):
        Omega = [0,0,1.]*u.one

        w0 = CartesianPhaseSpacePosition(pos=[1.,0,0.], vel=[0,1.,0.])

        # TODO: why doesn't this work when cython_if_possible=True
        orbit = self.obj.integrate_orbit(w0, dt=1., n_steps=1000,
                                         cython_if_possible=False,
                                         Integrator=DOPRI853Integrator)

        assert np.allclose(orbit.pos.value[0], 1., atol=1E-7)
        assert np.allclose(orbit.pos.value[1:], 0., atol=1E-7)

        # --------------------------------------------------------------
        # when Omega is off from orbital frequency
        #
        w0 = CartesianPhaseSpacePosition(pos=[1.,0,0.], vel=[0,1.1,0.])

        orbit = self.obj.integrate_orbit(w0, dt=0.1, n_steps=10000,
                                         cython_if_possible=False,
                                         Integrator=DOPRI853Integrator)

        rot_orbit = to_rotating_frame(Omega, orbit)

        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(1,2,figsize=(10,4.5))
        axes[0].plot(orbit.pos.value[0], orbit.pos.value[1], ls='none', alpha=0.4)
        axes[1].plot(rot_orbit.pos.value[0], rot_orbit.pos.value[1], ls='none', alpha=0.4)
        plt.show()

# class TestLogPotentialRotatingFrame(_TestBase):
#     obj = Hamiltonian(SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic),
#                       ConstantRotatingFrame(Omega=[0,0,20.]*u.km/u.s/u.kpc, units=galactic))

#     @pytest.mark.skip("Not implemented")
#     def test_hessian(self):
#         pass

#     def test_integrate(self):
#         import matplotlib.pyplot as plt

#         Omega = [0,0,20.]*u.km/u.s/u.kpc
#         w0 = CartesianPhaseSpacePosition(pos=[10.,0,0.]*u.kpc,
#                                          vel=[0,200,0.]*u.km/u.s)
#         rot_w0 = to_rotating_frame(Omega, w0, t=0*u.Myr)

#         orbit = self.obj.integrate_orbit(rot_w0, dt=1., n_steps=1000,
#                                          cython_if_possible=False,
#                                          Integrator=DOPRI853Integrator)
#         orbit.plot(linestyle='none', alpha=0.25)
#         plt.show()

#         return

#         # ---
#         H2 = Hamiltonian(SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic),
#                          StaticFrame(units=galactic))
#         w02 = CartesianPhaseSpacePosition(pos=[10.,0,0.]*u.kpc,
#                                           vel=[0,200,0.]*u.km/u.s)
#         orbit2 = H2.integrate_orbit(w02, dt=0.5, n_steps=10000)
#         orbit2.plot(linestyle='none', alpha=0.25)

#         rot_orbit = to_rotating_frame(-Omega, orbit2)
#         rot_orbit.plot(linestyle='none', alpha=0.25)

#         plt.show()
