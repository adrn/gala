"""LogarithmicPotential.density must use the same phi rotation as energy/gradient.

Public issue: https://github.com/adrn/gala/issues/633
"""

import astropy.units as u
import numpy as np

from gala.potential import LogarithmicPotential
from gala.units import galactic


def test_logarithmic_density_respects_phi():
    kwargs = dict(
        v_c=220 * u.km / u.s,
        r_h=12 * u.kpc,
        q1=1.3,
        q2=1.0,
        q3=0.8,
        units=galactic,
    )
    x = [3.5, 2.0, 1.0] * u.kpc
    pot0 = LogarithmicPotential(phi=0 * u.deg, **kwargs)
    pot20 = LogarithmicPotential(phi=20 * u.deg, **kwargs)

    assert pot0.energy(x) != pot20.energy(x)
    assert not np.allclose(pot0.gradient(x), pot20.gradient(x))

    d0 = pot0.density(x).to_value(u.Msun / u.kpc**3)
    d20 = pot20.density(x).to_value(u.Msun / u.kpc**3)
    # Unpatched C++ returns these equal; after rotating q by phi they differ.
    assert not np.allclose(d0, d20)

    # In-plane axisymmetry: q1==q2 makes phi a no-op for density.
    sym = dict(kwargs)
    sym["q1"] = 1.0
    sym["q2"] = 1.0
    s0 = LogarithmicPotential(phi=0 * u.deg, **sym)
    s20 = LogarithmicPotential(phi=20 * u.deg, **sym)
    assert u.allclose(s0.density(x), s20.density(x), rtol=1e-10)
