.. _actionangle:

****************************************************
Tutorial: Computing actions, angles, and frequencies
****************************************************

Introduction
============

Regular orbits permit a (local) transformation to a set of canonical coordinates
such that the momenta are independent, isolating integrals of motion (the actions,
:math:`\boldsymbol{J}`) and the conjugate coordinate variables (the angles,
:math:`\boldsymbol{\theta}`) linearly increase time. Action-angle coordinates are
useful for a number of applications because Hamilton's equations -- the equations
of motion -- are so simple:

.. math::

    H &= H(\boldsymbol{J})\\
    \dot{\boldsymbol{J}} &= -\frac{\partial H}{\partial \boldsymbol{\theta}} = 0\\
    \dot{\boldsymbol{\theta}} &= \frac{\partial H}{\partial \boldsymbol{J}} = \boldsymbol{\Omega}(\boldsymbol{J}) = {\rm constant}

Analytic transformations from phase-space to action-angle coordinates are only
known for a few simple cases where the gravitational potential is separable or
has many symmetries. However, astronomical systems can often be triaxial or
have complex radial profiles that are not captured by these simple systems.
Here we have implemented the method described in
:ref:`Sanders & Binney (2014) <references>` for computing actions and angles
for an arbitrary numerically integrated orbit. We test it below on three orbits:

* :ref:`a loop orbit in an axisymmetric potential <axisymmetric>`,
* :ref:`a loop orbit in a triaxial potential <triaxialloop>`,
* :ref:`an irregular orbit in the same triaxial potential <triaxialchaotic>`.

.. _axisymmetric:

Axisymmetric potential
----------------------

Triaxial potential
------------------

.. _triaxialloop:

Loop orbit
^^^^^^^^^^

.. _triaxialchaotic:

Irregular orbit
^^^^^^^^^^^^^^^

.. image:: ../_static/dynamics/orbit_Rz_loop.png

.. image:: ../_static/dynamics/action_hist_loop.png

.. image:: ../_static/dynamics/freq_hist_loop.png

.. image:: ../_static/dynamics/toy_computed_actions_loop.png

.. _references:

References
==========

* Binney & Tremaine (2008) `Galactic Dynamics <http://press.princeton.edu/titles/8697.html>`_
* Sanders & Binney (2014) `Actions, angles and frequencies for numerically integrated orbits <http://arxiv.org/abs/1401.3600>`_
* McGill & Binney (1990) `Torus construction in general gravitational potentials <http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1990MNRAS.244..634M&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf>`_