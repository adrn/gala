# coding: utf-8

# Third-party
import astropy.units as u
import numpy as np
import pytest

# Project
from gala._cconfig import GSL_ENABLED
from gala.units import galactic
from gala.potential.scf import SCFPotential, SCFInterpolatedPotential

if not GSL_ENABLED:
    pytest.skip("skipping SCF tests: they depend on GSL",
                allow_module_level=True)


@pytest.mark.parametrize('func_name', ['energy', 'density', 'gradient'])
def test_simple_compare_noninterp(func_name):
    """
    Compare the interpolated to time-invariant versions for a trivial case
    """
    rng = np.random.default_rng(42)
    nmax = 5
    lmax = 3

    Snlm = rng.uniform(size=(nmax+1, lmax+1, lmax+1))
    Tnlm = np.zeros_like(Snlm)

    tj = np.linspace(0, 1000, 16)
    Sjnlm = np.repeat(Snlm[None], len(tj), axis=0)
    Tjnlm = np.repeat(Tnlm[None], len(tj), axis=0)

    m = 1e9
    r_s = 10.

    pot_static = SCFPotential(m=m, r_s=r_s, Snlm=Snlm, Tnlm=Tnlm, units=galactic)
    pot_t = SCFInterpolatedPotential(
        m=m, r_s=r_s, tj=tj, Sjnlm=Sjnlm, Tjnlm=Tjnlm, units=galactic,
        com_xj=np.zeros((3, len(tj))), com_vj=np.zeros((3, len(tj)))
    )

    test_xyz = rng.uniform(-10, 10, size=(3, 10))
    test_t = rng.uniform(0, 1000, size=16)
    for t in test_t:
        t_val = getattr(pot_t, func_name)(test_xyz, t=t)
        static_val = getattr(pot_static, func_name)(test_xyz, t=t)
        assert u.allclose(t_val, static_val)
