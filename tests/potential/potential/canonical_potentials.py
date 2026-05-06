"""Instances of all builtin potential classes, for use in tests and benchmarks

A completeness check at the bottom of this module raises RuntimeError at
import time if any class in gala.potential.potential.builtin is neither
registered in CANONICAL nor listed in _EXCLUDED.
"""

import inspect

import astropy.units as u
from gala._cconfig import GSL_ENABLED

import gala.potential as gp
import gala.potential.potential.builtin as _builtin_module
from gala.potential.potential.core import PotentialBase
from gala.units import galactic, solarsystem

CANONICAL = {
    "HarmonicOscillatorPotential": gp.HarmonicOscillatorPotential(
        omega=[1.0, 1.0, 1.0]
    ),
    "HenonHeilesPotential": gp.HenonHeilesPotential(),
    "NullPotential": gp.NullPotential(),
    "KeplerPotential": gp.KeplerPotential(units=solarsystem, m=1.0),
    "IsochronePotential": gp.IsochronePotential(units=solarsystem, m=1.0, b=0.1),
    "HernquistPotential": gp.HernquistPotential(units=galactic, m=1.0e11, c=0.26),
    "PlummerPotential": gp.PlummerPotential(units=galactic, m=1.0e11, b=0.26),
    "JaffePotential": gp.JaffePotential(units=galactic, m=1.0e11, c=0.26),
    "MiyamotoNagaiPotential": gp.MiyamotoNagaiPotential(
        units=galactic, m=1.0e11, a=6.5, b=0.26
    ),
    "MN3ExponentialDiskPotential": gp.MN3ExponentialDiskPotential(
        units=galactic, m=1.0e11, h_R=3.5, h_z=0.26
    ),
    "SatohPotential": gp.SatohPotential(units=galactic, m=1.0e11, a=6.5, b=0.26),
    "KuzminPotential": gp.KuzminPotential(units=galactic, m=1.0e11, a=3.5),
    "StonePotential": gp.StonePotential(units=galactic, m=1e11, r_c=0.1, r_h=10.0),
    "PowerLawCutoffPotential": (
        gp.PowerLawCutoffPotential(units=galactic, m=1e10, r_c=10.0, alpha=1.8)
        if GSL_ENABLED
        else None
    ),
    "NFWPotential": gp.NFWPotential(units=galactic, m=1e11, r_s=12.0),
    "LeeSutoTriaxialNFWPotential": gp.LeeSutoTriaxialNFWPotential(
        units=galactic, v_c=0.35, r_s=12.0, a=1.3, b=1.0, c=0.8
    ),
    "LogarithmicPotential": gp.LogarithmicPotential(
        units=galactic, v_c=0.17, r_h=10.0, q1=1.2, q2=1.0, q3=0.8
    ),
    "LongMuraliBarPotential": gp.LongMuraliBarPotential(
        units=galactic, m=1e11, a=4.0, b=1, c=1.0
    ),
    "BurkertPotential": gp.BurkertPotential(
        units=galactic, rho=5e-25 * u.g / u.cm**3, r0=12
    ),
    "MultipolePotential": (
        gp.MultipolePotential(
            units=galactic, m=1e10, r_s=15.0, inner=True, lmax=2, S10=1.0, S21=0.5
        )
        if GSL_ENABLED
        else None
    ),
    "LM10Potential": gp.LM10Potential(),
    "MilkyWayPotential_v1": gp.MilkyWayPotential(version="v1"),
    "MilkyWayPotential_v2": gp.MilkyWayPotential(version="v2"),
    "BovyMWPotential2014": gp.BovyMWPotential2014() if GSL_ENABLED else None,
}

# Canonical potential class names for which density is not implemented.
# Used by the benchmark suite to skip the density test for these potentials.
NO_DENSITY = frozenset(
    {
        "HarmonicOscillatorPotential",
        "HenonHeilesPotential",
        "NullPotential",
        "KeplerPotential",
        "LM10Potential",
        "CylSplinePotential",
    }
)

# Potential classes that cannot have a simple parameter-only canonical instance.
# Most of these are used in specialized tests or benchmarks with custom setup.
_EXCLUDED = {
    "CylSplinePotential",
    "EXPPotential",
    "PyEXPPotential",
    "SphericalSplinePotential",
    "TimeInterpolatedPotential",
    "MilkyWayPotential",  # covered by MilkyWayPotential_v1 and MilkyWayPotential_v2
    "MilkyWayPotential2022",  # deprecated
}

# Check that all builtin classes are registered above (or explicitly excluded)
_all_builtin_classes = {
    name
    for name, obj in inspect.getmembers(_builtin_module, inspect.isclass)
    if issubclass(obj, PotentialBase) and obj is not PotentialBase
}

_unregistered = _all_builtin_classes - set(CANONICAL) - _EXCLUDED
if _unregistered:
    raise RuntimeError(
        "The following builtin potential classes are not registered in "
        "canonical_potentials.py. Add an entry to CANONICAL, or add the class "
        "name to _EXCLUDED with an explanation:\n"
        f"  {sorted(_unregistered)}"
    )
