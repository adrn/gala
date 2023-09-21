from .core import *
from .cpotential import *
from .ccompositepotential import *
from .builtin import *
from .io import *
from .util import *


def __getattr__(name):
    # Needed for MultipolePotential save/load
    from . import builtin

    if name in globals():
        return globals()[name]

    elif name.startswith('MultipolePotentialLmax'):
        return getattr(builtin.core, name)

    elif name.startswith('SCF'):
        from .. import scf
        return getattr(scf, name)

    else:
        raise AttributeError("huh")
