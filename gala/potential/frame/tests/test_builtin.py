# coding: utf-8

# Third-party
import astropy.units as u

# Project
from ..builtin import StaticFrame, ConstantRotatingFrame
from ...tests.helpers import _TestBase
from ....units import galactic

# class TestStaticFrame(_TestBase):
#     obj = StaticFrame(galactic)

# class TestConstantRotatingFrame(_TestBase):
#     obj = ConstantRotatingFrame(Omega=[1E-3,0.,0.]/u.Myr, units=galactic)

def test_init():
    fr1 = StaticFrame(galactic)
    fr2 = StaticFrame(galactic)
    assert fr1 == fr2

    fr1 = ConstantRotatingFrame(Omega=[1E-3,0.,0.]/u.Myr, units=galactic)
    fr2 = ConstantRotatingFrame(Omega=[1E-3,0.,0.]/u.Myr, units=galactic)
    fr3 = ConstantRotatingFrame(Omega=[2E-3,0.,0.]/u.Myr, units=galactic)
    assert fr1 == fr2
    assert fr1 != fr3

    st_fr = StaticFrame(galactic)
    assert st_fr != fr1
