# coding: utf-8
"""
    Test the core Potential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import pytest
import numpy as np
from astropy.constants import G
import astropy.units as u
import matplotlib.pyplot as plt

from ..core import Potential, CompositePotential

top_path = "/tmp/streamteam"
plot_path = os.path.join(top_path, "tests/potential")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

usys = [u.kpc,u.Myr,u.Msun,u.radian]

def test_simple():

    def f(r):
        return 1./r

    def f_prime(r):
        return r**-2

    p = Potential(f=f, f_prime=f_prime)
    assert p.value_at(0.5) == 2.
    assert p.acceleration_at(0.5) == 4.

    print(p.value_at(np.arange(0.5, 11.5, 0.5)))
    print(p.acceleration_at(np.arange(0.5, 11.5, 0.5)))

def test_simple_units():

    def f(r):
        return (G*222234404403.41818*u.Msun)/r

    def f_prime(r):
        return (G*222234404403.41818*u.Msun)/r**2

    p = Potential(f=f, f_prime=f_prime)
    assert np.allclose(p.value_at(0.5*u.kpc).decompose(usys), (2*u.kpc**2/u.Myr**2))
    assert np.allclose(p.acceleration_at(0.5*u.kpc).decompose(usys), (4*u.kpc/u.Myr**2))

def test_repr():

    def f(r,m):
        return G*m/r

    def f_prime(r,m):
        return -G*m/r**2

    p = Potential(f=f, f_prime=f_prime, parameters=dict(m=1.E10*u.Msun))
    assert p.__repr__() == "<Potential: m=1.00e+10 solMass>"

def test_composite():

    def f(x, m, x0):
        r = np.sqrt(np.sum((x-x0)**2, axis=-1))
        return G*m/r

    def f_prime(x, m, x0):
        r = np.sqrt(np.sum((x-x0)**2, axis=-1))
        return G*m*(x-x0)/r**3

    p1 = Potential(f=f, f_prime=f_prime,
                   parameters=dict(m=222234404403.41818*u.Msun,
                                   x0=np.array([[1.,0.,0.]])*u.kpc))

    p2 = Potential(f=f, f_prime=f_prime,
                   parameters=dict(m=222234404403.41818*u.Msun,
                                   x0=np.array([[-1.,0.,0.]])*u.kpc))

    p = CompositePotential(one=p1, two=p2)
    assert np.allclose(p.value_at([0.,0.,0.]*u.kpc).decompose(usys), 2*u.kpc**2/u.Myr**2)
    assert np.allclose(p.acceleration_at([0.,0.,0.]*u.kpc).decompose(usys).value, 0.)

    #fig, axes = p.plot_contours(grid=np.linspace(-10., 10., 100)*u.au)
    #fig.savefig(os.path.join(plot_path, "composite_point_mass.png"))

# def test_cartesian():

#     def f(x, m):
#         r = np.sqrt(np.sum(x**2, axis=-1))
#         return G*m/r

#     def f_prime(x, m):
#         r = np.sqrt(np.sum(x**2, axis=-1))
#         return -G*m/r**2

#     p = CartesianPotential(f=f, f_prime=f_prime, parameters=dict(m=1*u.Msun))
#     p.value_at(np.random.uniform(-10, 10, size=(100,3))*u.au)
#     p.acceleration_at(np.random.uniform(-10, 10, size=(100,3))*u.au)

#     fig, axes = p.plot_contours(grid=np.linspace(-10., 10., 100)*u.au)
#     fig.savefig(os.path.join(plot_path, "point_mass.png"))