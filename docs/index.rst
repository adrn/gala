.. include:: references.txt

.. _gala:

####
Gala
####

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

``gala`` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for ``gala`` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. ``gala`` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the ``Astropy`` package (`astropy.units` and
`astropy.coordinates`).


Most of the code in ``gala`` centers around implementations of
`gravitational potentials <potential/index.html>`_,
`numerical integration <integrate/index.html>`_,
and `nonlinear dynamics <dynamics/index.html>`_.
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
   benefits
   testing
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

   examples/Milky-Way-model.ipynb
   examples/integrate-potential-example
   examples/integrate-rotating-frame
   examples/mock-stream-heliocentric
   examples/circ-restricted-3body

*****************
How to contribute
*****************

We welcome contributions from anyone via pull requests on `GitHub
<https://github.com/adrn/gala>`_. If you don't feel comfortable modifying or
adding functionality, we also welcome feature requests and bug reports as
`GitHub issues <https://github.com/adrn/gala/issues>`_.

************
Contributors
************

.. include:: ../AUTHORS.rst
