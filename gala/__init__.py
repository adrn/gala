"""
Gala.
"""

from ._astropy_init import *

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from . import coordinates
    from . import dynamics
    from . import integrate
    from . import potential
    from . import units
    from . import util
    from . import mpl_style

    from .potential.builtin.special import stuff
    exec(stuff)

    # Monkey-patch Quantity
    import astropy.units as u

    @classmethod
    def _parse_quantity(cls, q):
        try:
            val,unit = q.split()
        except:
            val = q
            unit = u.dimensionless_unscaled

        return u.Quantity(float(val), unit)
    u.Quantity.from_string = _parse_quantity

    del u, stuff
