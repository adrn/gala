.. include:: ../references.txt

.. _gary-integrate:

************************************
Integration (`gary.integrate`)
************************************

Introduction
============

Scipy provides numerical ODE integration functions but they aren't very
user friendly or object oriented. This subpackage implements the Leapfrog
integration scheme (not available in Scipy) and provides wrappers to
higher order integration schemes such as a 5th order Runge-Kutta and the
Dormand-Prince 85(3) method.

>>> import astropy.units as u
>>> import numpy as np
>>> import gary.integrate as gi
>>> from gary.units import galactic

Getting Started
===============

All of the integrator classes have the same basic call structure. To create
an integrator object, you pass in a function that evaluates derivatives of,
for example, phase-space coordinates, then you call the `~gary.integrate.Integrator.run`
method while specifying timestep information. This is best seen with an example.

.. todo::

    Explain how to use units with the integrators...

Example: Forced pendulum
========================

Here we demonstrate how to use the Dormand-Prince integrator to compute the
orbit of a forced pendulum. We will use the variable ``q`` as the angle of the
pendulum with respect to the vertical and ``p`` as the conjugate momentum. Our
Hamiltonian is

.. math::

    H(q,p) = \frac{1}{2}p^2 + \cos(q) + A\sin(\omega_D t)

so that

.. math::

    \dot{q} &= p\\
    \dot{p} &= -\sin(q) + A\omega_D\cos(\omega_D t)

Our derivative function is then::

    >>> def F(t, w, A, omega_D):
    ...     q,p = w
    ...     q_dot = p
    ...     p_dot = -np.sin(q) + A*omega_D*np.cos(omega_D*t)
    ...     return np.array([q_dot, p_dot])

This function has two arguments -- :math:`A` (``A``), the amplitude of the forcing,
and :math:`\omega_D` (``omega_D``), the driving frequency. We define an integrator
object by specifying this function, along with values for the function arguments::

    >>> integrator = DOPRI853Integrator(F, func_args=(0.07, 0.75))

To integrate an orbit, we use the `~gary.integrate.Integrator.run` method.
We have to specify the initial conditions along with information about how long
to integrate and with what stepsize. There are several options for how to specify
the time step information. We could pre-generate an array of times and pass that in,
or pass in an initial time, end time, and timestep. Or, we could simply pass in the
number of steps to run for and a timestep. For this example, we will use the last
option. See the API below under *"Other Parameters"* for more information.::

    >>> orbit = integrator.run([3.,0.], dt=0.1, nsteps=10000)

We can plot the integrated (chaotic) orbit::

    >>> fig = orbit.plot()

.. plot::
    :align: center

    import astropy.units as u
    import matplotlib.pyplot as pl
    import numpy as np
    import gary.integrate as gi

    def F(t, w, A, omega_D):
        q,p = w
        q_dot = p
        p_dot = -np.sin(q) + A*omega_D*np.cos(omega_D*t)
        return np.array([q_dot, p_dot])

    integrator = gi.DOPRI853Integrator(F, func_args=(0.07, 0.75))
    orbit = integrator.run([3.,0.], dt=0.1, nsteps=10000)
    fig = orbit.plot()

Here's another example of integrating the
`Lorenz equations <https://en.wikipedia.org/wiki/Lorenz_system>`_, a 3D
nonlinear system::

    >>> def F(t,w,sigma,rho,beta):
    ...     x,y,z,px,py,pz = w
    ...     return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z, 0., 0., 0.]).reshape(w.shape)
    >>> sigma, rho, beta = 10., 28., 8/3.
    >>> integrator = Integrator(F, func_args=(sigma, rho, beta))
    >>> orbit = integrator.run([0.5,0.5,0.5,0,0,0], dt=1E-2, nsteps=1E4)
    >>> fig = orbit.plot()

.. plot::
    :align: center

    import astropy.units as u
    import matplotlib.pyplot as pl
    import numpy as np
    import gary.integrate as gi

    def F(t,w,sigma,rho,beta):
        x,y,z,px,py,pz = w
        return np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z, 0., 0., 0.]).reshape(w.shape)

    sigma, rho, beta = 10., 28., 8/3.
    integrator = Integrator(F, func_args=(sigma, rho, beta))

    orbit = integrator.run([0.5,0.5,0.5,0,0,0], dt=1E-2, nsteps=1E4)
    fig = orbit.plot()

API
===

.. automodapi:: gary.integrate
