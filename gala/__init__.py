"""
Gala.
"""

__author__ = 'adrn <adrianmpw@gmail.com>'

from ._astropy_init import *

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from . import coordinates
    from . import dynamics
    from . import integrate
    from . import potential
    from . import units
    from . import util
    # from . import mpl_style
