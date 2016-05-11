# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import pytest

# Custom
from ....potential import SphericalNFWPotential
from ....dynamics import CartesianPhaseSpacePosition
from ....integrate import DOPRI853Integrator
from ....units import galactic

# Project
from ..core import mock_stream, streakline_stream, fardal_stream, dissolved_fardal_stream

def test_mock_stream():
    potential = SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic)

    w0 = CartesianPhaseSpacePosition(pos=[0.,15.,0]*u.kpc,
                                     vel=[-0.13,0,0]*u.kpc/u.Myr)
    prog = potential.integrate_orbit(w0, dt=-2., n_steps=1023)
    prog = prog[::-1]

    k_mean = [1.,0.,0.,0.,1.,0.]
    k_disp = [0.,0.,0.,0.,0.,0.]
    stream = mock_stream(potential, prog, k_mean=k_mean, k_disp=k_disp,
                         prog_mass=1E4, Integrator=DOPRI853Integrator)

    # fig = prog.plot(subplots_kwargs=dict(sharex=False,sharey=False))
    # fig = stream.plot(color='#ff0000', alpha=0.5, axes=fig.axes)
    # fig = stream.plot()
    # pl.show()

    assert stream.pos.shape == (3,2048) # two particles per step

    diff = np.abs(stream[-2:].pos - prog[-1].pos)
    assert np.allclose(diff[0].value, 0.)
    assert np.allclose(diff[1,0].value, diff[1,1].value)
    assert np.allclose(diff[2].value, 0.)

mock_funcs = [streakline_stream, fardal_stream, dissolved_fardal_stream]
all_extra_args = [dict(), dict(), dict(t_disrupt=-250.*u.Myr)]
@pytest.mark.parametrize("mock_func, extra_kwargs", zip(mock_funcs, all_extra_args))
def test_each_type(mock_func, extra_kwargs):
    potential = SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic)

    w0 = CartesianPhaseSpacePosition(pos=[0.,15.,0]*u.kpc,
                                     vel=[-0.13,0,0]*u.kpc/u.Myr)
    prog = potential.integrate_orbit(w0, dt=-2., n_steps=1023)
    prog = prog[::-1]

    stream = mock_func(potential, prog_orbit=prog, prog_mass=1E4,
                       Integrator=DOPRI853Integrator, **extra_kwargs)

    # fig = prog.plot(subplots_kwargs=dict(sharex=False,sharey=False))
    # fig = stream.plot(color='#ff0000', alpha=0.5, axes=fig.axes)
    # fig = stream.plot()
    # pl.show()

    assert prog.t.shape == (1024,)
    assert stream.pos.shape == (3,2048) # two particles per step
