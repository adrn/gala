# Third-party
import astropy.units as u
import pytest

# Custom
from ....potential import Hamiltonian, NFWPotential
from ....dynamics import PhaseSpacePosition
from ....integrate import DOPRI853Integrator
from ....units import galactic

# Project
from ..core import (mock_stream, streakline_stream, fardal_stream)


def test_mock_stream():
    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.,
                                                    units=galactic)
    ham = Hamiltonian(potential)

    w0 = PhaseSpacePosition(pos=[0., 15., 0]*u.kpc,
                            vel=[-0.13, 0, 0]*u.kpc/u.Myr)
    prog = ham.integrate_orbit(w0, dt=-2., n_steps=1023)
    prog = prog[::-1]

    k_mean = [1., 0. ,0. ,0. ,1., 0.]
    k_disp = [0., 0., 0., 0., 0., 0.]

    with pytest.raises(NotImplementedError):
        mock_stream(ham, prog, k_mean=k_mean, k_disp=k_disp, prog_mass=1E4)


mock_funcs = [streakline_stream, fardal_stream]
all_extra_args = [dict(), dict()]
@pytest.mark.parametrize("mock_func, extra_kwargs", zip(mock_funcs, all_extra_args))
def test_each_type(mock_func, extra_kwargs):
    # TODO: remove this test when these functions are removed
    potential = NFWPotential.from_circular_velocity(v_c=0.2, r_s=20.,
                                                    units=galactic)
    ham = Hamiltonian(potential)

    w0 = PhaseSpacePosition(pos=[0., 15., 0]*u.kpc,
                            vel=[-0.13, 0, 0]*u.kpc/u.Myr)
    prog = ham.integrate_orbit(w0, dt=-2., n_steps=1023)
    prog = prog[::-1]

    with pytest.warns(DeprecationWarning):
        stream = mock_func(ham, prog_orbit=prog, prog_mass=1E4*u.Msun,
                           Integrator=DOPRI853Integrator, **extra_kwargs)

    assert prog.t.shape == (1024,)
    assert stream.pos.shape == (2048,) # two particles per step
