.. include:: references.txt

.. _nd-representations:

************************************
N-dimensional representation classes
************************************

Introduction
============

The Astropy |astropyrep| presently only support 3D positions and differential
objects.

TODO: astropy only supports 3D...


    >>> fig = w.plot(marker='o', s=40, alpha=0.5)

.. plot::
    :align: center

    import astropy.units as u
    import numpy as np
    import gala.dynamics as gd
    np.random.seed(42)
    x = np.random.uniform(-10,10,size=(3,128))
    v = np.random.uniform(-200,200,size=(3,128))
    w = gd.CartesianPhaseSpacePosition(pos=x*u.kpc,
                                       vel=v*u.km/u.s)
    fig = w.plot(marker='o', s=40, alpha=0.5)


N-dimensional representations API
---------------------------------
.. automodapi:: gala.dynamics.representation_nd
    :no-heading:
    :headings: ^^
