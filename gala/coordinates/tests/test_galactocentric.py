# Third-party
import astropy.coordinates as coord

# This package
from ..galactocentric import get_galactocentric2019


def test_simple():
    # Not meant to be a sensible test - that's done in Astropy!
    fr = get_galactocentric2019()
    assert isinstance(fr, coord.Galactocentric)
