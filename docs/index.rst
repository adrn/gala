.. include:: references.txt

.. _gala:

####
Gala
####

Gala is a Python package for Galactic astronomy and gravitational
dynamics. The bulk of the package centers around implementations of
`gravitational potentials <potential/index>`_,
`numerical integration <integrate/index>`_,
and `nonlinear dynamics <dynamics/index>`_.

The package is being actively developed in
`a public repository on GitHub <https://github.com/adrn/gala>`_ so if you
have any trouble,
`open an issue <https://github.com/adrn/gala/issues>`_ there.

*************
Documentation
*************

.. toctree::
   :maxdepth: 1

   install
   conventions
   why
..   getting_started

***********
Subpackages
***********

.. toctree::
   :maxdepth: 1

   coordinates/index
   integrate/index
   potential/index
   dynamics/index
   units
   util

*********
Tutorials
*********

.. toctree::
   :maxdepth: 1
   :glob:

   examples/integrate-potential-example
