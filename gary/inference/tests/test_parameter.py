# coding: utf-8

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import pytest
import astropy.units as u

from ..parameter import *
from ..prior import *

class TestModelParameter(object):

    def test_string(self):
        p = ModelParameter("m", truth=1.5, prior=UniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm'>"
        assert str(p) == "m"

        p = ModelParameter("m", prior=UniformPrior(1.,2.))
        assert repr(p) == "<ModelParameter 'm'>"
        assert str(p) == "m"

    def test_dumb(self):
        p = ModelParameter("test")
        assert p.name == "test"
        assert p.frozen is False

    def test_truthvalue(self):
        p = ModelParameter("test", truth=15.)
        assert p.truth.value == 15.
        assert p.truth.unit == u.dimensionless_unscaled

        p = ModelParameter("test", truth=15.*u.km)
        assert p.truth.value == 15.
        assert p.truth.unit == u.km

        p = ModelParameter("test", truth=np.ones(15)*21.*u.km)
        assert p.truth.shape == (15,)
        assert p.truth.unit == u.km

        p = ModelParameter("test", shape=(15,))
        assert p.truth.shape == (15,)
        assert p.truth.unit == u.dimensionless_unscaled

        # one with units, one without not allowed
        with pytest.raises(ValueError):
            p = ModelParameter("test", truth=11*u.km, shape=(15,))

    def test_prior(self):
        prior = UniformPrior(a=10., b=25.)
        p = ModelParameter("test", truth=21., prior=prior)
