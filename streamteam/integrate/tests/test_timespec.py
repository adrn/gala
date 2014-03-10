# coding: utf-8
"""
    Test the time specification parser.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys

# Third-party
import numpy as np
import pytest

# Project
from ..timespec import _parse_time_specification

def test_time_parse():

    # dt, nsteps[, t1] : (numeric, int[, numeric])
    t = _parse_time_specification(dt=0.1, nsteps=100)
    np.testing.assert_allclose(np.min(t), 0.)
    np.testing.assert_allclose(np.max(t), 10.-0.1)
    assert len(t) == 100

    t = _parse_time_specification(dt=0.1, nsteps=100, t1=10.)
    np.testing.assert_allclose(np.min(t), 10.)
    np.testing.assert_allclose(np.max(t), 20.-0.1)
    assert len(t) == 100

    # dt, t1, t2 : (numeric, numeric, numeric)
    t = _parse_time_specification(dt=0.1, t1=10., t2=130.)
    np.testing.assert_allclose(np.min(t), 10.)
    np.testing.assert_allclose(np.max(t), 130.)

    t = _parse_time_specification(dt=-0.1, t1=10., t2=-13.412)
    np.testing.assert_allclose(np.min(t), -13.412)
    np.testing.assert_allclose(np.max(t), 10.)

    with pytest.raises(ValueError):
        _parse_time_specification(dt=-0.1, t1=10., t2=130.)

    with pytest.raises(ValueError):
        _parse_time_specification(dt=0.1, t1=130., t2=-10.142)

    # dt, t1 : (array_like, numeric)
    t = _parse_time_specification(dt=np.random.random(size=100), t1=31.412)
    np.testing.assert_allclose(np.min(t), 31.412)

    # nsteps, t1, t2 : (int, numeric, numeric)
    t = _parse_time_specification(nsteps=100, t1=24.124, t2=91.412)
    np.testing.assert_allclose(np.min(t), 24.124)
    np.testing.assert_allclose(np.max(t), 91.412)

    t = _parse_time_specification(nsteps=100, t1=24.124, t2=-91.412)
    np.testing.assert_allclose(np.max(t), 24.124)
    np.testing.assert_allclose(np.min(t), -91.412)

    # t : array_like
    input = np.arange(0., 10., 0.1)
    t = _parse_time_specification(t=input)
    assert (t == input).all()