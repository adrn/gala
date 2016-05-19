.. _define-new-potential:

*********************************
Defining your own potential class
*********************************

Introduction
============

There are two ways to define a new potential class: with pure-Python, or with
Cython. The advantage to writing a new class in Cython is that the
computations can execute with C-like speeds, however only certain integrators
support using this functionality anyways. If you are not familiar with Cython,
you probably want to stick to a pure-Python class.

For code blocks below and any pages linked below, I assume the following
imports have already been excuted::

    >>> import numpy as np
    >>> import gala.potential as gp
    >>> import gala.dynamics as gd

Implementing a new potential with Python
========================================

New potentials are implemented by subclassing one of the potential base
classes and defining functions that compute the value and gradient of the
potential. We will work through an example below for adding a
`Henon-Heiles potential <http://en.wikipedia.org/wiki/H%C3%A9non-Heiles_System>`_.

The expression for the potential is:

.. math::

    \Phi(x,y) = \frac{1}{2}(x^2 + y^2) + A\,(x^2 y - \frac{y^3}{3})

With this parametrization, there is only one free parameter, and the potential
is two-dimensional.

At minimum, the subclass must implement:

- ``__init__()``
- ``_value()``

For integration, it must also implement a ``_gradient()`` method.

The ``_value`` method will compute the value of the potential at a given
position and time (e.g., the potential energy). The ``_gradient`` computes
the gradient of the potential. Both of these methods must accept two arguments:
a position, and a time. Because this potential has a free parameter, our
``__init__`` method must accept a parameter argument and store this in the
``self.parameters`` dictionary (a required attribute of any subclass).
Let's write it out, then work through what each piece means in detail::

    >>> class HenonHeilesPotential(gp.PotentialBase):
    ...    def __init__(self, A, units):
    ...        pars = dict(A=A)
    ...        super(HenonHeilesPotential, self).__init__(units=units,
    ...                                                   parameters=pars)
    ...
    ...    def _value(self, q, t):
    ...        A = self.parameters['A']
    ...        x,y = q
    ...        return 0.5*(x**2 + y**2) + A*(x**2*y - y**3/3)
    ...
    ...    def _gradient(self, q, t):
    ...        A = self.parameters['A']
    ...        x,y = q
    ...
    ...        grad = np.zeros_like(q)
    ...        grad[0] = x + 2*A*x*y
    ...        grad[1] = y + A*(x**2 - y**2)
    ...
    ...        return grad

The value and gradient methods simply compute the numerical value and
gradient of the potential. The ``__init__`` method must take a single
argument, ``A``, and store this to a paremeter dictionary. We also provide
no default unit system so that this is also a required argument of the class.
To initialize the class, we would have to pass in these two arguments.
For now, let's pass in ``None`` for the unit system to designate that we'll
work with dimensionless quantities::

    >>> pot = HenonHeilesPotential(A=1., units=None)

That's it! Now we have a fully-fledged potential object. For example, we
can integrate an orbit in this potential::

    >>> w0 = gd.CartesianPhaseSpacePosition(pos=[0.,0.3],
    ...                                     vel=[0.38,0.])
    >>> orbit = pot.integrate_orbit(w0, dt=0.05, n_steps=10000)
    >>> fig = orbit.plot()

.. plot::
    :align: center

    import matplotlib.pyplot as pl
    import numpy as np
    import gala.dynamics as gd
    import gala.potential as gp

    class HenonHeilesPotential(gp.PotentialBase):

        def __init__(self, A, units):
            pars = dict(A=A)
            super(HenonHeilesPotential, self).__init__(units=units,
                                                       parameters=pars)

        def _value(self, q, t):
            A = self.parameters['A']
            x,y = q
            return 0.5*(x**2 + y**2) + A*(x**2*y - y**3/3)

        def _gradient(self, q, t):
            A = self.parameters['A']
            x,y = q
            print(x)
            grad = np.zeros_like(q)
            grad[0] = x + 2*A*x*y
            grad[1] = y + A*(x**2 - y**2)
            return grad

    pot = HenonHeilesPotential(A=1., units=None)
    w0 = gd.CartesianPhaseSpacePosition(pos=[0.,0.3],
                                        vel=[0.38,0.])
    orbit = pot.integrate_orbit(w0, dt=0.05, n_steps=10000)
    fig = orbit.plot()

Or, we could create a contour plot of equipotentials::

    >>> grid = np.linspace(-1.5,1.5,100)
    >>> fig = pot.plot_contours(grid=(grid,grid),
    ...                         levels=[0, 0.05,0.1,1/6.,0.5,1.,2,3,5],
    ...                         cmap='Blues_r')

.. plot::
    :align: center

    import matplotlib.pyplot as pl
    import numpy as np
    import gala.dynamics as gd
    import gala.potential as gp

    class HenonHeilesPotential(gp.PotentialBase):

        def __init__(self, A, units):
            pars = dict(A=A)
            super(HenonHeilesPotential, self).__init__(units=units,
                                                       parameters=pars)

        def _value(self, q, t):
            A = self.parameters['A']
            x,y = q
            return 0.5*(x**2 + y**2) + A*(x**2*y - y**3/3)

        def _gradient(self, q, t):
            A = self.parameters['A']
            x,y = q
            print(x)
            grad = np.zeros_like(q)
            grad[0] = x + 2*A*x*y
            grad[1] = y + A*(x**2 - y**2)
            return grad

    pot = HenonHeilesPotential(A=1., units=None)
    grid = np.linspace(-1.5,1.5,100)
    fig = pot.plot_contours(grid=(grid,grid), cmap='Blues_r', levels=[0, 0.05,0.1,1/6.,0.5,1.,2,3,5])

Adding a custom potential with Cython
-------------------------------------

.. todo::

    Need to write this.
