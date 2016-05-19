__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import pytest

# Project
from ..util import ImmutableDict

def test_immutabledict():
    a = dict(a=5, c=6)
    b = ImmutableDict(**a)

    with pytest.raises(TypeError):
        b['test'] = 5
