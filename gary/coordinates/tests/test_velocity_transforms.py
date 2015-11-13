# coding: utf-8

""" Test velocity transformations.  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging

# Third-party
import numpy as np
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u

# This project
from ..velocity_transforms import *

logger.setLevel(logging.DEBUG)

class TestTransforms(object):

    def setup(self):
        self.pos = ([[10.,0,0], [10,0,0], [4,4,0]] * u.kpc).T
        self.vel = ([[0.,100,500], [0,-100,-500], [100.,100.,-500]] * u.km/u.s).T

        self.pos_repr = coord.CartesianRepresentation(self.pos)

    def test_cartesian_to(self):
        for i,func in enumerate([cartesian_to_spherical,
                                 cartesian_to_physicsspherical,
                                 cartesian_to_cylindrical]):

            # dimensionless
            vsph1 = func(self.pos.value * u.dimensionless_unscaled,
                         self.vel.value * u.dimensionless_unscaled)
            assert vsph1.unit == u.dimensionless_unscaled

            # astropy coordinates
            cpos = coord.SkyCoord(self.pos_repr)
            vsph2 = func(cpos, self.vel)
            assert vsph2.unit == u.km/u.s

            # astropy representation
            vsph3 = func(self.pos_repr, self.vel)
            assert vsph3.unit == u.km/u.s

            np.testing.assert_allclose(vsph1.value, vsph2.value, atol=1E-10)
            np.testing.assert_allclose(vsph1.value, vsph3.value, atol=1E-10)

            true_v = self.vel.copy()
            if func == cartesian_to_physicsspherical:
                true_v[2] *= -1.

            np.testing.assert_allclose(vsph1[:,:2].value, true_v[:,:2].value)

            assert vsph1[0,2] > 0.  # vr
            assert np.sign(vsph1[2,2]) == np.sign(true_v[2,2])
            np.testing.assert_allclose(np.sqrt(np.sum(vsph1**2,axis=0)).value,
                                       np.sqrt(np.sum(true_v**2,axis=0)).value)

    def test_to_cartesian(self):
        for i,func in enumerate([spherical_to_cartesian,
                                 physicsspherical_to_cartesian,
                                 cylindrical_to_cartesian]):
            true_v = ([[0.,100,500], [0,-100,-500], [0.,np.sqrt(2)*100,-500]] * u.km/u.s).T

            # dimensionless
            vsph1 = func(self.pos.value * u.dimensionless_unscaled,
                         self.vel.value * u.dimensionless_unscaled)
            assert vsph1.unit == u.dimensionless_unscaled

            # astropy coordinates
            cpos = coord.SkyCoord(self.pos_repr)
            vsph2 = func(cpos, self.vel)
            assert vsph2.unit == u.km/u.s

            # astropy representation
            vsph3 = func(self.pos_repr, self.vel)
            assert vsph3.unit == u.km/u.s

            np.testing.assert_allclose(vsph1.value, vsph2.value, atol=1E-10)
            np.testing.assert_allclose(vsph1.value, vsph3.value, atol=1E-10)

            if func == physicsspherical_to_cartesian:
                true_v[2] *= -1.

            np.testing.assert_allclose(vsph1.value, true_v.value, atol=1E-10)
