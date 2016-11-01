# coding: utf-8

# Third-party
import astropy.units as u

# Project
from ..builtin import StaticFrame, ConstantRotatingFrame
from ...tests.helpers import _TestBase
from ....units import galactic

class TestStaticFrame(_TestBase):
    obj = StaticFrame(galactic)

class TestConstantRotatingFrame(_TestBase):
    obj = ConstantRotatingFrame(Omega=[1E-3,0.,0.]/u.Myr, units=galactic)
