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