.. _scf:

Self-consistent field (SCF)
===========================

``gala.scf`` contains utilities for evaluating basis function expansions of mass
densities and gravitational potentials with the Self-Consistent Field (SCF)
method of Hernquist & Ostriker (1992; [HO92]_). SCF uses Hernquist radial
functions and spherical harmonics for angular functions. This implementation is
based on the formalism described in the original paper but using the notation of
Lowing et al. (2011; [L11]_).

.. raw:: html

   <video controls src="../_static/anim-prof.mp4" width=450 height=320 autoplay loop></video>


Introduction
------------

The two main ways to use `gala.potential.scf` are:

#. to compute the expansion coefficients given a continuous density distribution
   or discrete samples from a density distribution, then
#. to evaluate the density, potential, and gradients of a basis function
   expansion representation of a density distribution given this set of
   coefficients.


To compute expansion coefficients, the relevant functions are
`~gala.potential.scf.compute_coeffs` and
`~gala.potential.scf.compute_coeffs_discrete`. This implementation uses the
notation from [L11]_: all expansion coefficients are real, :math:`S_{nlm}` are
the cosine coefficients, and :math:`T_{nlm}` are the sine coefficients.

Once you have coefficients, there are two ways to evaluate properties of the
potential or the density of the expansion representation. `gala` provides a
class-based interface :class:`~gala.potential.scf.SCFPotential` that utilizes
the gravitational potential machinery implemented in `gala.potential` (and
supports all of the standard potential functionality, such as orbit integration
and plotting). The examples below use this interface.

Examples
--------
- :ref:`coeff-particle`
- :ref:`coeff-analytic`
- :ref:`potential-class`

.. toctree::
  :hidden:

  scf-examples

API
---

.. automodapi:: gala.potential.scf


----------
References
----------
.. [HO92] http://dx.doi.org/10.1086/171025
.. [L11] http://dx.doi.org/10.1111/j.1365-2966.2011.19222.x
