# coding: utf-8

""" Test utilities  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

# Project
from ..util import peak_to_peak_period, estimate_dt_n_steps
from ...potential import SphericalNFWPotential
from ...units import galactic

def test_peak_to_peak_period():
    ntimes = 16384

    # trivial test
    for true_T in [1., 2., 4.123]:
        t = np.linspace(0,10.,ntimes)
        f = np.sin(2*np.pi/true_T * t)
        T = peak_to_peak_period(t, f)
        assert np.allclose(T, true_T, atol=1E-3)

    # modulated trivial test
    true_T = 2.
    t = np.linspace(0,10.,ntimes)
    f = np.sin(2*np.pi/true_T * t) + 0.1*np.cos(2*np.pi/(10*true_T) * t)
    T = peak_to_peak_period(t, f)
    assert np.allclose(T, true_T, atol=1E-3)

def test_estimate_dt_n_steps():
    nperiods = 128
    pot = SphericalNFWPotential(v_c=1., r_s=10., units=galactic)
    w0 = [10.,0.,0.,0.,0.9,0.]
    dt,n_steps = estimate_dt_n_steps(w0, pot, n_periods=nperiods, n_steps_per_period=256,
                                     func=np.nanmin)

    orbit = pot.integrate_orbit(w0, dt=dt, n_steps=n_steps)
    T = orbit.estimate_period()
    assert int(round((orbit.t.max()/T).decompose().value)) == nperiods
