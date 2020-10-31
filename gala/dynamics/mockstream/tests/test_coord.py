"""
Note:
    This is just a way to get pytest to call tests implemented in Cython!
    See _coord.pyx for the actual test functions.
"""

# Project
from .._coord import (_test_sat_rotation_matrix, _test_to_sat_coords_roundtrip,
                      _test_car_to_cyl_roundtrip, _test_cyl_to_car_roundtrip)


def test_sat_rotation_matrix():
    _test_sat_rotation_matrix()


def test_to_sat_coords_roundtrip():
    _test_to_sat_coords_roundtrip()


def test_car_to_cyl_roundtrip():
    _test_car_to_cyl_roundtrip()


def test_cyl_to_car_roundtrip():
    _test_cyl_to_car_roundtrip()
