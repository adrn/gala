# Third-party
import astropy.units as u
import numpy as np
import pytest

# Custom
from ..core import MockStream


def test_init():

    xyz = np.random.random(size=(3, 100)) * u.kpc
    vxyz = np.random.random(size=(3, 100)) * u.km / u.s
    t1 = np.random.random(size=100) * u.Myr

    lead_trail = np.empty(100, dtype='U1')
    lead_trail[::2] = 't'
    lead_trail[1::2] = 'l'

    stream = MockStream(xyz, vxyz)
    stream = MockStream(xyz, vxyz, release_time=t1)
    stream = MockStream(xyz, vxyz, lead_trail=lead_trail)

    with pytest.raises(ValueError):
        MockStream(xyz, vxyz, release_time=t1[:-1])

    with pytest.raises(ValueError):
        MockStream(xyz, vxyz, lead_trail=lead_trail[:-1])
