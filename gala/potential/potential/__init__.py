from .core import *
from .cpotential import *
from .ccompositepotential import *
from .builtin import *
from .io import *
from .util import *
from .interop import *


def __getattr__(name):
    # Needed for MultipolePotential save/load
    from . import builtin

    if name in globals():
        return globals()[name]

    elif name.startswith('MultipolePotentialLmax'):
        return getattr(builtin.core, name)

    else:
        raise AttributeError("huh")
