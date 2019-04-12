# Third-party
import astropy.units as u
import numpy as np
import pytest

# This package
from .. import (GD1Koposov10, MagellanicStreamNidever08, OphiuchusPriceWhelan16,
                OrphanKoposov19, OrphanNewberg10, Pal5PriceWhelan18,
                SagittariusLaw10)

stream_frames = (
    GD1Koposov10, MagellanicStreamNidever08, OphiuchusPriceWhelan16,
    OrphanKoposov19, OrphanNewberg10, Pal5PriceWhelan18, SagittariusLaw10)


@pytest.mark.parametrize("frame_cls", stream_frames)
def test_wrapping(frame_cls):
    c = frame_cls([-60, 300]*u.deg, [15, -15]*u.deg)
    lon_name = list(c.get_representation_component_names().keys())[0]
    lat_name = list(c.get_representation_component_names().keys())[1]
    assert np.allclose(getattr(c, lon_name).value, -60)

    # with velocity data:
    data = dict()
    data['pm_{}_cos{}'.format(lon_name, lat_name)] = [1., 2.] * u.mas/u.yr
    data['pm_{}'.format(lat_name)] = [1., 2.] * u.mas/u.yr
    c = frame_cls([-60, 300]*u.deg, [15, -15]*u.deg, **data)
    assert np.allclose(getattr(c, lon_name).value, -60)
