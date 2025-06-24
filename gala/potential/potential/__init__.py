from .builtin import *
from .ccompositepotential import *
from .core import *
from .cpotential import *
from .io import *
from .util import *


def __getattr__(name):
    # Needed for MultipolePotential save/load
    from . import builtin

    if name in globals():
        return globals()[name]

    if name.startswith("MultipolePotentialLmax"):
        return getattr(builtin.core, name)

    if name.startswith("SCF"):
        from .. import scf

        return getattr(scf, name)

    raise AttributeError("huh")
