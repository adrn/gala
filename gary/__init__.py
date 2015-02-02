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

from . import coordinates
from . import dynamics
from . import integrate
from . import io
from . import potential

# Add a custom log level
from astropy import log as logger
import logging

def debug_factory(logger, debug_level):
    def custom_debug(msg, *args, **kwargs):
        if logger.level >= debug_level:
           return
        logger._log(debug_level, msg, args, kwargs)
    return custom_debug

logging.addLevelName(logging.INFO+5, 'IMPORTANT')
setattr(logger, 'important', debug_factory(logger, logging.INFO+5))
logger.setLevel(logging.INFO+1)

from potential.custom import stuff
exec(stuff)

del logging, debug_factory, u
