"""
    Test conversions in core.py
"""

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np

# This package
from ..velocity_frame_transforms import vgsr_to_vhel, vhel_to_vgsr


def test_vgsr_to_vhel():
    filename = get_pkg_data_filename('idl_vgsr_vhel.txt')
    data = np.genfromtxt(filename, names=True, skip_header=2)

    # one row
    row = data[0]
    l = coord.Angle(row["lon"] * u.degree)
    b = coord.Angle(row["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vgsr = row["vgsr"] * u.km/u.s
    vlsr = [row["vx"], row["vy"], row["vz"]] * u.km/u.s  # this is right
    vcirc = row["vcirc"] * u.km/u.s

    vsun = vlsr + [0, 1, 0]*vcirc
    vhel = vgsr_to_vhel(c, vgsr, vsun=vsun)
    assert np.allclose(vhel.value, row['vhelio'], atol=1e-3)

    # now check still get right answer passing in ICRS coordinates
    vhel = vgsr_to_vhel(c.transform_to(coord.ICRS()), vgsr, vsun=vsun)
    assert np.allclose(vhel.value, row['vhelio'], atol=1e-3)

    # all together now
    l = coord.Angle(data["lon"] * u.degree)
    b = coord.Angle(data["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vgsr = data["vgsr"] * u.km/u.s
    vhel = vgsr_to_vhel(c, vgsr, vsun=vsun)
    assert np.allclose(vhel.value, data['vhelio'], atol=1e-3)

    # now check still get right answer passing in ICRS coordinates
    vhel = vgsr_to_vhel(c.transform_to(coord.ICRS()), vgsr, vsun=vsun)
    assert np.allclose(vhel.value, data['vhelio'], atol=1e-3)


def test_vgsr_to_vhel_misc():
    # make sure it works with longitude in 0-360 or -180-180
    l1 = coord.Angle(190.*u.deg)
    l2 = coord.Angle(-170.*u.deg)
    b = coord.Angle(30.*u.deg)

    c1 = coord.Galactic(l1, b)
    c2 = coord.Galactic(l2, b)

    vgsr = -110.*u.km/u.s
    vhel1 = vgsr_to_vhel(c1, vgsr)
    vhel2 = vgsr_to_vhel(c2, vgsr)

    assert np.allclose(vhel1.value, vhel2.value)


def test_vhel_to_vgsr():
    filename = get_pkg_data_filename('idl_vgsr_vhel.txt')
    data = np.genfromtxt(filename, names=True, skip_header=2)

    # one row
    row = data[0]
    l = coord.Angle(row["lon"] * u.degree)
    b = coord.Angle(row["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vhel = row["vhelio"] * u.km/u.s
    vlsr = [row["vx"], row["vy"], row["vz"]] * u.km/u.s  # this is right
    vcirc = row["vcirc"]*u.km/u.s

    vsun = vlsr + [0, 1, 0] * vcirc
    vgsr = vhel_to_vgsr(c, vhel, vsun=vsun)
    assert np.allclose(vgsr.value, row['vgsr'], atol=1e-3)

    # now check still get right answer passing in ICRS coordinates
    vgsr = vhel_to_vgsr(c.transform_to(coord.ICRS()), vhel, vsun=vsun)
    assert np.allclose(vgsr.value, row['vgsr'], atol=1e-3)

    # all together now
    l = coord.Angle(data["lon"] * u.degree)
    b = coord.Angle(data["lat"] * u.degree)
    c = coord.Galactic(l, b)
    vhel = data["vhelio"] * u.km/u.s
    vgsr = vhel_to_vgsr(c, vhel, vsun=vsun)
    assert np.allclose(vgsr.value, data['vgsr'], atol=1e-3)

    # now check still get right answer passing in ICRS coordinates
    vgsr = vhel_to_vgsr(c.transform_to(coord.ICRS()), vhel, vsun=vsun)
    assert np.allclose(vgsr.value, data['vgsr'], atol=1e-3)
