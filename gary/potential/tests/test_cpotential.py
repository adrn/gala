# coding: utf-8
"""
    Test the core Potential classes
"""

from __future__ import absolute_import, unicode_literals, division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import os
import time
import numpy as np
from astropy.utils.console import color_print

from ..core import CompositePotential
from ..cpotential import CCompositePotential
from ...potential import cbuiltin as pot
from ...units import galactic

top_path = "plots/"
plot_path = os.path.join(top_path, "tests/potential")
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

print()
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")
color_print("To view plots:", "green")
print("    open {}".format(plot_path))
color_print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", "yellow")

def test_time():
    potential1 = pot.KeplerPotential(m=1E11, units=galactic)
    potential2 = pot.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.26, units=galactic)
    comp_potential = CCompositePotential(one=potential1, two=potential2)

    q = np.random.uniform(-10, 10, size=(10,3))
    energy = potential1.value(q) + potential2.value(q)
    comp_energy = comp_potential.value(q)
    np.testing.assert_allclose(comp_energy, energy)

    for n in [10000]:
        q = np.random.uniform(-10, 10, size=(n,3))

        t1 = time.time()
        for i in range(1000):
            energy = potential1.value(q) + potential2.value(q)
        print("{} orbits -- Python: {:.1f}".format(n, time.time()-t1))

        t1 = time.time()
        for i in range(1000):
            comp_energy = comp_potential.value(q)
        print("{} orbits -- Cython: {:.1f}".format(n, time.time()-t1))

        t1 = time.time()
        for i in range(1000):
            comp_energy = potential1.value(q)
        print("{} orbits -- Single potential: {:.1f}".format(n, time.time()-t1))
