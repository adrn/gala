.. include:: references.txt

.. _gala:

.. raw:: html

   <img src="_static/Gala_Logo_RGB.png" width="50%"
    style="margin-bottom: 32px;"/>

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

.. ---------------------
.. Nav bar (top of docs)

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:

   install
   getting_started
   tutorials
   reference
   contributing
   glossary


************
Contributors
************

.. include:: ../AUTHORS.rst

************************
Citation and Attribution
************************

|JOSS| |DOI|

`Here is a list of papers that use Gala
<https://ui.adsabs.harvard.edu/search/q=citations(bibcode%3A2017JOSS....2..388P)&sort=date%20desc%2C%20bibcode%20desc&p_=0>`_

If you make use of this code, please cite the `JOSS <http://joss.theoj.org>`_
paper:

.. code-block:: bibtex

    @article{gala,
      doi = {10.21105/joss.00388},
      url = {https://doi.org/10.21105%2Fjoss.00388},
      year = 2017,
      month = {oct},
      publisher = {The Open Journal},
      volume = {2},
      number = {18},
      author = {Adrian M. Price-Whelan},
      title = {Gala: A Python package for galactic dynamics},
      journal = {The Journal of Open Source Software}

Please consider also citing the Zenodo DOI |DOI| of the version you used as a
software citation:

.. include:: ZENODO.rst

.. |JOSS| image:: http://joss.theoj.org/papers/10.21105/joss.00388/status.svg
   :target: http://joss.theoj.org/papers/10.21105/joss.00388
.. |DOI| image:: https://zenodo.org/badge/17577779.svg
   :target: https://zenodo.org/badge/latestdoi/17577779
