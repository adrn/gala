.. _integrate:

************************************
Integration (`streamteam.integrate`)
************************************

Introduction
============

Scipy provides numerical ODE integration functions but they aren't very
user friendly or object oriented. This subpackage implements the Leapfrog
integration scheme (not available in Scipy) and provides wrappers to
higher order integration schemes such as a 5th order Runge-Kutta and the
Dormand-Prince 85(3) method.

Getting Started
===============

All of the integrator classes have the same basic call structure. To create
an integrator object, you pass in a function that evaluates derivatives of,
for example, phase-space coordinates, then you call the `.run()` method while
specifying timestep information. This is best seen with an example.

Example: Forced pendulum
========================

Here we demonstrate how to use the Dormand-Prince integrator to compute the
orbit of a forced pendulum. We will use the variable `q` as the angle of the
pendulum with respect to the vertical and `p` as the conjugate momentum. Our
Hamiltonian is

.. math::

    H(q,p) = \frac{1}{2}p^2 + \cos(q) + A\sin(\omega_D t)

so that

.. math::

    \dot{q} &= p\\
    \dot{p} &= -\sin(q) + A\omega_D\cos(\omega_D t)

Our derivative function is then::

    def F(t, w, A, omega_D):
        q,p = w.T
        q_dot = p
        p_dot = -np.sin(q) + A*omega_D*np.cos(omega_D*t)
        return np.array([q_dot, p_dot]).T

This function has two arguments -- :math:`A` (`A`), the amplitude of the forcing,
and :math:`\omega_D` (`omega_D`), the driving frequency. We define an integrator
object by specifying this function, along with values for the function arguments::

    integrator = DOPRI853Integrator(F, func_args=(0.07, 0.75))

To integrate an orbit, we use the `.run()` method. We have to specify the initial
conditions along with information about how long to integrate and with what
stepsize. There are several options for how to specify the time step information.
We could pre-generate an array of times and pass that in, or pass in an initial
time, end time, and timestep. Or, we could simply pass in the number of steps to
run for and a timestep. For this example, we will use the last option. See the
API below under *"Other Parameters"* for more information.::

    times,qps = integrator.run([3.,0.], dt=0.1, nsteps=100)
    q = qps[:,0,0]
    p = qps[:,0,1]

To demonstrate, we then plot the integrated (chaotic) orbit. The code that
generates this figure is shown below the plot.

.. image:: ../_static/integrate/forced-pendulum.png

::

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # create a figure
    plt.figure(figsize=(10,10))

    # define a 2x2 grid
    gs = GridSpec(2,2)

    # the first subplot is the whole top row, q vs p
    plt.subplot(gs[0,:])
    plt.plot(q, p, marker=None)
    plt.xlabel("$q$")
    plt.ylabel("$p$")

    plt.subplot(gs[1,0])
    plt.plot(times, q, marker=None)
    plt.xlabel("time step")
    plt.ylabel("$q$")
    plt.xlim(min(times),max(times))

    plt.subplot(gs[1,1])
    plt.plot(times, p, marker=None)
    plt.xlabel("time step")
    plt.ylabel("$p$")
    plt.xlim(min(times),max(times))


Reference/API
=============
.. autoclass:: streamteam.integrate.DOPRI853Integrator
   :members: run
.. autoclass:: streamteam.integrate.LeapfrogIntegrator
   :members: run
.. autoclass:: streamteam.integrate.RK5Integrator
   :members: run