import astropy.units as u
import numpy as np
import pytest

from .. import (
    GD1Koposov10,
    MagellanicStreamNidever08,
    OphiuchusPriceWhelan16,
    OrphanKoposov19,
    OrphanNewberg10,
    Pal5PriceWhelan18,
    SagittariusLaw10,
)

stream_frames = (
    GD1Koposov10,
    MagellanicStreamNidever08,
    OphiuchusPriceWhelan16,
    OrphanKoposov19,
    OrphanNewberg10,
    Pal5PriceWhelan18,
    SagittariusLaw10,
)


@pytest.mark.parametrize("frame_cls", stream_frames)
def test_wrapping(frame_cls):
    c = frame_cls([-60, 300] * u.deg, [15, -15] * u.deg)
    lon_name = next(iter(c.get_representation_component_names().keys()))
    lat_name = list(c.get_representation_component_names().keys())[1]
    assert np.allclose(getattr(c, lon_name).value, -60)

    # with velocity data:
    data = {}
    data[f"pm_{lon_name}_cos{lat_name}"] = [1.0, 2.0] * u.mas / u.yr
    data[f"pm_{lat_name}"] = [1.0, 2.0] * u.mas / u.yr
    c = frame_cls([-60, 300] * u.deg, [15, -15] * u.deg, **data)
    assert np.allclose(getattr(c, lon_name).value, -60)
