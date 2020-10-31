# Third-party
import astropy.units as u
import pytest
import numpy as np

# Project
from .helpers import _TestBase
from .. import Hamiltonian
from ...potential.builtin import NFWPotential, KeplerPotential, HernquistPotential
from ...frame.builtin import StaticFrame, ConstantRotatingFrame
from ....units import galactic, dimensionless
from ....dynamics import PhaseSpacePosition, Orbit
from ....integrate import DOPRI853Integrator

# ----------------------------------------------------------------------------


def to_rotating_frame(omega, w, t=None):
    """
    TODO: figure out units shit for omega and t
    TODO: move this to be a ConstantRotatingFrame method
    """

    if not hasattr(omega, 'unit'):
        raise TypeError("Input frequency vector must be a Quantity object.")

    try:
        omega = omega.to(u.rad/u.Myr, equivalencies=u.dimensionless_angles()).value
    except:  # noqa
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
        t = np.atleast_1d(t)  # works with Quantity's
    else:
        t = w.t

    try:
        t = t.to(u.Myr).value
    except:  # noqa
        t = t.value

    if isinstance(w, PhaseSpacePosition) or isinstance(w, Orbit):
        Cls = w.__class__
        x_shape = w.xyz.shape
        x_unit = w.x.unit
        v_unit = w.v_x.unit

        x = w.xyz.reshape(3, -1).value
        v = w.v_xyz.reshape(3, -1).value

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
        + (1 - np.cos(theta)) * np.einsum("i, ij->j", ee, x) * ee[:, None]

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


class TestWithPotentialStaticFrame(_TestBase):
    obj = Hamiltonian(NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.,
                                                          units=galactic),
                      StaticFrame(units=galactic))

    @pytest.mark.skip("Not implemented")
    def test_hessian(self):
        pass


class TestKeplerRotatingFrame(_TestBase):
    Omega = [0., 0, 1.]*u.one
    E_unit = u.one
    obj = Hamiltonian(KeplerPotential(m=1., units=dimensionless),
                      ConstantRotatingFrame(Omega=Omega, units=dimensionless))

    @pytest.mark.skip("Not implemented")
    def test_hessian(self):
        pass

    def test_integrate(self):

        w0 = PhaseSpacePosition(pos=[1., 0, 0.], vel=[0, 1., 0.])

        for bl in [True, False]:
            orbit = self.obj.integrate_orbit(w0, dt=1., n_steps=1000,
                                             cython_if_possible=bl,
                                             Integrator=DOPRI853Integrator)

            assert np.allclose(orbit.x.value, 1., atol=1E-7)
            assert np.allclose(orbit.xyz.value[1:], 0., atol=1E-7)


class TestKepler2RotatingFrame(_TestBase):
    Omega = [1., 1., 1.]*u.one
    E_unit = u.one
    obj = Hamiltonian(KeplerPotential(m=1., units=dimensionless),
                      ConstantRotatingFrame(Omega=Omega, units=dimensionless))

    @pytest.mark.skip("Not implemented")
    def test_hessian(self):
        pass

    def test_integrate(self):

        # --------------------------------------------------------------
        # when Omega is off from orbital frequency
        #
        w0 = PhaseSpacePosition(pos=[1., 0, 0.], vel=[0, 1.1, 0.])

        for bl in [True, False]:
            orbit = self.obj.integrate_orbit(w0, dt=0.1, n_steps=10000,
                                             cython_if_possible=bl,
                                             Integrator=DOPRI853Integrator)

            L = orbit.angular_momentum()
            C = orbit.energy() - np.sum(self.Omega[:, None] * L, axis=0)
            dC = np.abs((C[1:]-C[0])/C[0])
            assert np.all(dC < 1E-9)  # conserve Jacobi constant


@pytest.mark.parametrize("name, Omega, tol", [
    ("z-aligned co-rotating", [0, 0, 1.]*u.one, 1E-12),
    ("z-aligned", [0, 0, 1.5834]*u.one, 1E-12),
    ("random", [0.95792653, 0.82760659, 0.66443135]*u.one, 1E-10),
])
def test_velocity_rot_frame(name, Omega, tol):
    # _i = inertial
    # _r = rotating

    r0 = 1.245246
    potential = HernquistPotential(m=1., c=0.2, units=dimensionless)
    vc = potential.circular_velocity([r0, 0, 0]).value[0]
    w0 = PhaseSpacePosition(pos=[r0, 0, 0.],
                            vel=[0, vc, 0.])
    Omega = Omega * [1., 1., vc/r0]

    H_r = Hamiltonian(potential, ConstantRotatingFrame(Omega=Omega, units=dimensionless))
    H = Hamiltonian(potential, StaticFrame(units=dimensionless))

    orbit_i = H.integrate_orbit(w0, dt=0.1, n_steps=1000, Integrator=DOPRI853Integrator)
    orbit_r = H_r.integrate_orbit(w0, dt=0.1, n_steps=1000, Integrator=DOPRI853Integrator)

    orbit_i2r = orbit_i.to_frame(ConstantRotatingFrame(Omega=Omega, units=dimensionless))
    orbit_r2i = orbit_r.to_frame(StaticFrame(units=dimensionless))

    assert u.allclose(orbit_i.xyz, orbit_r2i.xyz, atol=tol)
    assert u.allclose(orbit_i.v_xyz, orbit_r2i.v_xyz, atol=tol)

    assert u.allclose(orbit_r.xyz, orbit_i2r.xyz, atol=tol)
    assert u.allclose(orbit_r.v_xyz, orbit_i2r.v_xyz, atol=tol)
