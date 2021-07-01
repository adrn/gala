.. include:: references.txt

.. _nd-representations:

************************************
N-dimensional representation classes
************************************

For the examples below the following imports have already been executed::

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import gala.dynamics as gd

Introduction
============

The Astropy |astropyrep|_ presently only support 3D positions and differential
objects. The `~gala.dynamics.representation_nd.NDCartesianRepresentation` and
`~gala.dynamics.representation_nd.NDCartesianDifferential` classes add Cartesian
representation classes that can handle arbitrary numbers of dimensions. For
example, 2D coordinates::

    >>> xy = np.arange(16).reshape(2, 8) * u.kpc
    >>> rep = gd.NDCartesianRepresentation(xy)
    >>> rep
    <NDCartesianRepresentation (x1, x2) in kpc
        [(0.,  8.), (1.,  9.), (2., 10.), (3., 11.), (4., 12.), (5., 13.),
         (6., 14.), (7., 15.)]>

4D coordinates::

    >>> x = np.arange(16).reshape(4, 4) * u.kpc
    >>> rep = gd.NDCartesianRepresentation(x)
    >>> rep
    <NDCartesianRepresentation (x1, x2, x3, x4) in kpc
        [(0., 4.,  8., 12.), (1., 5.,  9., 13.), (2., 6., 10., 14.),
         (3., 7., 11., 15.)]>

These can be passed in to the |psp| or |orb| classes as with any of the Astropy
core representation objects::

    >>> xy = np.arange(16).reshape(2, 8) * u.kpc
    >>> vxy = np.arange(16).reshape(2, 8) / 10. * u.kpc / u.Myr
    >>> w = gd.PhaseSpacePosition(pos=xy, vel=vxy)
    >>> fig = w.plot()

.. plot::
    :align: center
    :width: 60%

    import astropy.units as u
    import numpy as np
    import gala.dynamics as gd
    xy = np.arange(16).reshape(2, 8) * u.kpc
    vxy = np.arange(16).reshape(2, 8) / 10. * u.kpc / u.Myr
    w = gd.PhaseSpacePosition(pos=xy, vel=vxy)
    fig = w.plot()

However, certain functionality such as representation transformations, dynamical
quantity calculation, and coordinate frame transformations are disabled when the
number of dimensions is not 3 (i.e. when not using the Astropy core
representation classes).

N-dimensional representations API
---------------------------------
.. automodapi:: gala.dynamics.representation_nd
    :no-heading:
    :headings: ^^
