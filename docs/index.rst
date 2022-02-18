.. include:: references.txt

.. raw:: html

   <img src="_static/Gala_Logo_RGB.png" width="50%"
    style="margin-bottom: 32px;"/>

.. module:: gala

****
Gala
****

Galactic Dynamics is the study of the formation, history, and evolution of
galaxies using the *orbits* of objects — numerically-integrated trajectories of
stars, dark matter particles, star clusters, or galaxies themselves.

``gala`` is an Astropy-affiliated Python package that aims to provide efficient
tools for performing common tasks needed in Galactic Dynamics research. Much of
this code uses Python for flexible, user-friendly interfaces that interact with
wrappers around low-level code (primarily C) to enable fast computations.
Common operations include `gravitational potential and force evaluations
<potential/index.html>`_, `orbit integrations <integrate/index.html>`_,
`dynamical coordinate transformations <dynamics/index.html>`_, and computing
`chaos indicators for nonlinear dynamics <dynamics/nonlinear.html>`_. ``gala``
heavily uses the units and astronomical coordinate systems defined in the
Astropy core package (:ref:`astropy.units <astropy-units>` and
:ref:`astropy.coordinates <astropy-coordinates>`).

This package is being actively developed in `a public repository on GitHub
<https://github.com/adrn/gala>`_, and we are always looking for new
contributors! No contribution is too small, so if you have any trouble with this
code, find a typo, or have requests for new content (tutorials or features),
please `open an issue on GitHub <https://github.com/adrn/gala/issues>`_.

.. note::

   Welcome to the new Gala documentation! The documentation has a new theme and
   some new content. Use the links in the nav bar above, the page listing below,
   or the search bar on the left to dive in!

.. ---------------------
.. Nav bar (top of docs)

.. toctree::
   :maxdepth: 1
   :titlesonly:

   install
   getting_started
   tutorials
   user_guide
   contributing


Contributors
============

.. include:: ../AUTHORS.rst


Citation and Attribution
========================

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
      journal = {The Journal of Open Source Software}}

Please also cite the Zenodo DOI |DOI| of the version you used as a software
citation:

.. include:: ZENODO.rst

.. |JOSS| image:: http://joss.theoj.org/papers/10.21105/joss.00388/status.svg
   :target: http://joss.theoj.org/papers/10.21105/joss.00388
.. |DOI| image:: https://zenodo.org/badge/17577779.svg
   :target: https://zenodo.org/badge/latestdoi/17577779
