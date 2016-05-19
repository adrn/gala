# coding: utf-8
"""
    Test the time specification parser.
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np
import pytest

# Project
from ..timespec import parse_time_specification
from ...units import DimensionlessUnitSystem, galactic

def test_dt_n_steps():
    # dt, n_steps[, t1] : (numeric, int[, numeric])
    t = parse_time_specification(DimensionlessUnitSystem(), dt=0.1, n_steps=100)
    np.testing.assert_allclose(np.min(t), 0.)
    np.testing.assert_allclose(np.max(t), 10.)
    assert len(t) == 101

    t = parse_time_specification(DimensionlessUnitSystem(), dt=0.1, n_steps=100, t1=10.)
    np.testing.assert_allclose(np.min(t), 10.)
    np.testing.assert_allclose(np.max(t), 20.)
    assert len(t) == 101

def test_dt_t1_t2():
    # dt, t1, t2 : (numeric, numeric, numeric)
    t = parse_time_specification(DimensionlessUnitSystem(), dt=0.1, t1=10., t2=130.)
    np.testing.assert_allclose(np.min(t), 10.)
    np.testing.assert_allclose(np.max(t), 130.)

    t = parse_time_specification(DimensionlessUnitSystem(), dt=-0.1, t1=10., t2=-13.412)
    np.testing.assert_allclose(np.min(t), -13.412)
    np.testing.assert_allclose(np.max(t), 10.)

    with pytest.raises(ValueError):
        parse_time_specification(DimensionlessUnitSystem(), dt=-0.1, t1=10., t2=130.)

    with pytest.raises(ValueError):
        parse_time_specification(DimensionlessUnitSystem(), dt=0.1, t1=130., t2=-10.142)

def test_n_steps_t1_t2():
    # n_steps, t1, t2 : (int, numeric, numeric)
    t = parse_time_specification(DimensionlessUnitSystem(), n_steps=100, t1=24.124, t2=91.412)
    np.testing.assert_allclose(np.min(t), 24.124)
    np.testing.assert_allclose(np.max(t), 91.412)

    t = parse_time_specification(DimensionlessUnitSystem(), n_steps=100, t1=24.124, t2=-91.412)
    np.testing.assert_allclose(np.max(t), 24.124)
    np.testing.assert_allclose(np.min(t), -91.412)

def test_t():
    # t : array_like
    input = np.arange(0., 10., 0.1)
    t = parse_time_specification(DimensionlessUnitSystem(), t=input)
    assert (t == input).all()

def test_t_units():
    # t : array_like
    input = np.linspace(0., 10., 100)*u.Gyr
    t = parse_time_specification(galactic, t=input)

    assert np.allclose(t[-1], 10000.)
