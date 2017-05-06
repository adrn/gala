.. _define-new-potential:

*********************************
Defining your own potential class
*********************************

Introduction
============

There are two ways to define a new potential class: with pure-Python, or with C
and Cython. The advantage to writing a new class in Cython is that the
computations can execute with C-like speeds, however only certain integrators
support using this functionality (Leapfrog and DOP853) and it is a bit more
complicated to set up the code to build the C+Cython code properly. If you are
not familiar with Cython, you probably want to stick to a pure-Python class for
initial testing. If there is a potential class that you think should be
included, feel free to suggest the new addition as a `GitHub issue
<https://github.com/adrn/gala/issues>`_!

For code blocks below and any pages linked below, I assume the following
imports have already been excuted::

    >>> import numpy as np
    >>> import gala.potential as gp
    >>> import gala.dynamics as gd

Implementing a new potential with Python
========================================

New Python potentials are implemented by subclassing
:class:`~gala.potential.potential.PotentialBase` and defining functions that
compute (at minimum) the energy and gradient of the potential. We will work
through an example below for adding the `Henon-Heiles potential
<http://en.wikipedia.org/wiki/H%C3%A9non-Heiles_System>`_.

The expression for the potential is:

.. math::

    \Phi(x,y) = \frac{1}{2}(x^2 + y^2) + A\,(x^2 y - \frac{y^3}{3})

With this parametrization, there is only one free parameter (``A``), and the
potential is two-dimensional.

At minimum, the subclass must implement:

- ``__init__()``
- ``_energy()``
- ``_gradient()``

The ``_energy()`` method will compute the potential energy at a given position
and time. The ``_gradient()`` computes the gradient of the potential. Both of
these methods must accept two arguments: a position, and a time. These internal
methods are then called by the :class:`~gala.potential.potential.PotentialBase`
superclass methods :meth:`~gala.potential.potential.PotentialBase.energy` and
:meth:`~gala.potential.potential.PotentialBase.gradient`. The superclass methods
convert the input position to an array in the unit system of the potetial for
fast evalutaion. The input to these superclass methods can be
:class:`~astropy.units.Quantity` objects,
:class:`~gala.dynamics.PhaseSpacePosition` objects, or :class:`~numpy.ndarray`.

Because this potential has a free parameter, the ``__init__`` method must accept
a parameter argument and store this in the ``parameters`` dictionary attribute
(a required attribute of any subclass). Let's write it out, then work through
what each piece means in detail::

    >>> class HenonHeilesPotential(gp.PotentialBase):
    ...
    ...     def __init__(self, A, units=None):
    ...         pars = dict(A=A)
    ...         super(HenonHeilesPotential, self).__init__(units=units,
    ...                                                    parameters=pars,
    ...                                                    ndim=2)
    ...
    ...     def _energy(self, xy, t):
    ...         A = self.parameters['A'].value
    ...         x,y = xy.T
    ...         return 0.5*(x**2 + y**2) + A*(x**2*y - y**3/3)
    ...
    ...     def _gradient(self, xy, t):
    ...         A = self.parameters['A'].value
    ...         x,y = xy.T
    ...
    ...         grad = np.zeros_like(xy)
    ...         grad[:,0] = x + 2*A*x*y
    ...         grad[:,1] = y + A*(x**2 - y**2)
    ...         return grad

The internal energy and gradient methods compute the numerical value and
gradient of the potential. The ``__init__`` method must take a single argument,
``A``, and store this to a paremeter dictionary. The expected shape of the
position array (``xy``) passed to the internal ``_energy()`` and ``_gradient()``
methods is always 2-dimensional with shape ``(n_points, n_dim)`` where
``n_points >= 1`` and ``n_dim`` must match the dimensionality of the potential
specified in the initializer. Note that this is different from the shape
expected when calling the actual methods ``energy()`` and ``gradient()``!

Let's now create an instance of the class and see how it works. For now, let's
pass in ``None`` for the unit system to designate that we'll work with
dimensionless quantities::

    >>> pot = HenonHeilesPotential(A=1., units=None)

That's it! Now we have a fully-fledged potential object. For example, we
can integrate an orbit in this potential::

    >>> w0 = gd.PhaseSpacePosition(pos=[0.,0.3],
    ...                            vel=[0.38,0.])
    >>> orbit = pot.integrate_orbit(w0, dt=0.05, n_steps=10000)
    >>> fig = orbit.plot(marker=',', linestyle='none', alpha=0.5)

.. plot::
    :align: center

    import matplotlib.pyplot as pl
    import numpy as np
    import gala.dynamics as gd
    import gala.potential as gp

    class HenonHeilesPotential(gp.PotentialBase):

        def __init__(self, A, units=None):
            pars = dict(A=A)
            super(HenonHeilesPotential, self).__init__(units=units,
                                                       parameters=pars,
                                                       ndim=2)

        def _energy(self, q, t):
            A = self.parameters['A'].value
            x,y = q.T
            return 0.5*(x**2 + y**2) + A*(x**2*y - y**3/3)

        def _gradient(self, q, t):
            A = self.parameters['A'].value
            x,y = q.T

            grad = np.zeros_like(q)
            grad[:,0] = x + 2*A*x*y
            grad[:,1] = y + A*(x**2 - y**2)
            return grad

    pot = HenonHeilesPotential(A=1., units=None)
    w0 = gd.PhaseSpacePosition(pos=[0.,0.3],
                               vel=[0.38,0.])
    orbit = pot.integrate_orbit(w0, dt=0.05, n_steps=10000)
    fig = orbit.plot(marker=',', linestyle='none', alpha=0.5)

Or, we could create a contour plot of equipotentials::

    >>> grid = np.linspace(-1., 1., 100)
    >>> from matplotlib import colors
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1, figsize=(5,5))
    >>> fig = pot.plot_contours(grid=(grid,grid),
    ...                         levels=np.logspace(-3, 1, 10),
    ...                         norm=colors.LogNorm(),
    ...                         cmap='Blues', ax=ax)

.. plot::
    :align: center

    from matplotlib import colors
    import matplotlib.pyplot as pl
    import numpy as np
    import gala.dynamics as gd
    import gala.potential as gp

    class HenonHeilesPotential(gp.PotentialBase):

        def __init__(self, A, units=None):
            pars = dict(A=A)
            super(HenonHeilesPotential, self).__init__(units=units,
                                                       parameters=pars,
                                                       ndim=2)

        def _energy(self, q, t):
            A = self.parameters['A'].value
            x,y = q.T
            return 0.5*(x**2 + y**2) + A*(x**2*y - y**3/3)

        def _gradient(self, q, t):
            A = self.parameters['A'].value
            x,y = q.T

            grad = np.zeros_like(q)
            grad[:,0] = x + 2*A*x*y
            grad[:,1] = y + A*(x**2 - y**2)
            return grad

    pot = HenonHeilesPotential(A=1., units=None)
    grid = np.linspace(-1.,1.,100)
    fig,ax = pl.subplots(1, 1, figsize=(5,5))
    fig = pot.plot_contours(grid=(grid,grid), cmap='Blues',
                            levels=np.logspace(-3, 1, 10),
                            norm=colors.LogNorm(), ax=ax)

Adding a custom potential with Cython
-------------------------------------

.. todo::

    More info coming soon. For now, contact `adrn <https://github.com/adrn>`_
    for help.
