.. _exp_tutorial:

==============================
Using EXP potentials with Gala
==============================

Gala supports `EXP <https://exp-docs.readthedocs.io>`_ as a backend for representing
flexible and time-dependent gravitational potentials, typically constructed from N-body
simulation snapshots. This requires:

#. building EXP,
#. building Gala with EXP support,
#. and setting up a `~gala.potential.potential.EXPPotential` object using the user's EXP config and
   coefficient files.

Note that EXP support currently requires building Gala from source.
Additionally, this workflow has only been tested on Linux and MacOS with the setups seen
in the `GitHub actions test config file
<https://github.com/adrn/gala/blob/main/.github/workflows/tests.yml>`_.

------------
Building EXP
------------

The `EXP documentation <https://exp-docs.readthedocs.io/en/latest/intro/install.html>`_
is the best place to read about how to build EXP. Gala doesn't have any special
requirements for the EXP build, except that the user must actually "install" EXP,
rather than just build it. This is demonstrated below.

To install EXP's dependencies, here is one recipe that we have found to work on Ubuntu 24.04::

    sudo apt-get install build-essential cmake gfortran git libeigen3-dev libfftw3-dev libhdf5-dev libomp-dev libopenmpi-dev ninja-build
    # install uv python, only needed if you don't already have python:
    # curl -LsSf https://astral.sh/uv/install.sh | sh

Here is another recipe using modules that has been found to work on Flatiron Institute's rusty cluster::

    module load modules/2.4 cmake gcc openmpi hdf5 libtirpc eigen fftw git python uv

EXP also builds on Mac by installing the dependencies with Homebrew::

    brew install cmake eigen fftw hdf5 open-mpi git ninja

After installing the dependencies, one can download and build EXP on Linux with::

    git clone --recursive https://github.com/EXP-code/EXP.git
    cd EXP
    cmake -G Ninja -B build --install-prefix $PWD/install
    cmake --build build
    cmake --install build

In this case, we installed EXP to the ``EXP/install/`` directory, but this can be any
directory. This will become the ``GALA_EXP_PREFIX`` directory in the next step.

For a full example of how to build EXP on Mac, see `this build recipe
<https://gist.github.com/adrn/afd9222416e359fcef826b7988b7d69f>`_.

Note that building pyEXP is not required. However, some tests will use pyEXP if it is
present.

------------------------------
Building Gala with EXP support
------------------------------

Building Gala with the ``GALA_EXP_PREFIX`` environment variable set to the EXP install dir
will trigger compilation of the Gala's EXP Cython extensions. For example::

    git clone https://github.com/adrn/gala.git
    cd gala
    export GALA_EXP_PREFIX=/path/to/EXP/install/

If you build and install EXP following the instructions above, the EXP installation will be
located in ``EXP/install/``. If you installed EXP to a different location, you can set the
``GALA_EXP_PREFIX`` to that location. In either case, ``GALA_EXP_PREFIX`` must be the directory
that contains the subdirectories ``lib`` and ``include``.

Now you can run the Gala build. For example, using uv::

    uv venv
    uv pip install -ve .

Or using venv::

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -ve .

In either case, the output should show a message like ``Gala: installing with EXP support``.

Note that in previous versions of Gala, the ``GALA_EXP_PREFIX`` was supposed to point to the
EXP repo root, rather than the EXP installation directory. This is no longer the case. The
EXP repo and build directories are not needed to build Gala with EXP support.

Likewise, ``GALA_EXP_LIB_PATH`` was used in past Gala versions but not anymore.


----------------------------------
Running Gala with an EXP potential
----------------------------------

To use an EXP potential with Gala, first you'll need a config file, a basis file, and a
coefficients file from EXP. We have included example files with this tutorial, produced
by constructing a basis and computing coefficients with particle data from a single
snapshot of the dark matter halo of the m12m simulation in the `Latte suite
<https://fire.northwestern.edu/latte/>`_ of the `FIRE-2 simulations
<https://arxiv.org/abs/1702.06148>`_. In particular, the relevant files are:

- ``m12m-basis.yml`` - the basis configuration file
- ``m12m_basis_table.model`` - the basis table (density and potential evaluated on a
  grid of spherical radii)
- ``m12m-coef.hdf5`` - the coefficients file

The basis was generated with a unit system in which G=1 (standard for EXP), the mass
unit is :math:`10^{12}~\mathrm{M}_\odot`, and the length unit is 10 kpc.
Setting up an `~gala.potential.potential.EXPPotential` object with these files is as easy as
specifying the unit system and EXP files:

.. code-block:: python

    import astropy.units as u
    import gala.potential as gp
    from gala.units import SimulationUnitSystem

    exp_units = SimulationUnitSystem(mass=1e12 * u.Msun, length=10 * u.kpc, G=1)

    exp_pot = gp.EXPPotential(
        units=exp_units,
        config_file="data/m12m-basis.yml",
        coef_file="data/m12m-coef.hdf5",
    )

Then one can use the potential object like any other Gala potential. For example, to
integrate and plot an orbit:

.. code-block:: python

    import gala.dynamics as gd

    w0 = gd.PhaseSpacePosition(
        pos=[8, 0.0, 1.0] * u.kpc,
        vel=[0.0, 220, 0.0] * u.km / u.s,
    )
    orbit = gp.Hamiltonian(exp_pot).integrate_orbit(w0, dt=1 * u.Myr, t1=0, t2=6 * u.Gyr)
    fig = orbit.plot(units=u.kpc, linestyle="-", alpha=0.5, label="orbit in m12m")

-----
Units
-----

Gala generally works in physical units (e.g., kpc, solar mass, etc.), whereas EXP
typically works in user-defined simulation units. To use EXP with Gala, one must define
a `~gala.units.SimulationUnitSystem` and specify this when creating the potential (as
demonstrated above). If the basis was computed from a scale-dependent potential, the
simulation unit system must match the units used to generate the basis. If the potential
was computed from a scale-independent model, the simulation unit system can be
arbitrary, but it can be used to set physical scales to the simulations.

--------------
Time Evolution
--------------

An `~gala.potential.potential.EXPPotential` may be time-evolving or static. If the coefficient
file has only one snapshot, the potential will be static. Likewise, if ``tmin``/``tmax``
are passed such that only one snapshot from the coefs falls within that range, the
potential will be static. For the examples below, we use hypothetical files
``config.yml`` and ``coefs.h5`` that contain coefficients for multiple snapshots.

One can always check if an ``EXPPotential`` is static with:

.. code-block:: python

    exp_pot.static

One can also "freeze" make a multi-snapshot potential (i.e. make it static) by selecting
a single snapshot with the ``snapshot_index`` parameter:

.. code-block:: python

    exp_pot = gp.EXPPotential(
        units=exp_units,
        config_file="config.yml",
        coef_file="coefs.h5",
        snapshot_index=0,
    )

For time-evolving potentials, if one tries to evaluate the potential outside of the
time range stored in the coefficients file (even indirectly, such as during an
orbit integration), a C++ exception will be triggered, which will be raised to the user
as a Python exception. The Python exception will contain the error message from C++.
For example:
``RuntimeError: FieldWrapper::interpolator: time t=11.73 is out of bounds: [0.0195404, 11.724]``.

If the coefficients file stores a very large time range but the user is only interested
in a smaller range, one can specify ``tmin`` and/or ``tmax`` to load a smaller subset of
the coefficient data (for memory efficiency):

.. code-block:: python

    exp_pot = gp.EXPPotential(
        units=exp_units,
        config_file="config.yml",
        coef_file="coefs.h5",
        tmin=1.0,
        tmax=2.0,
    )

Note that, as mentioned above, subsequently using a time outside this range will result
in a Python exception. Or more precisely: using a time outside the range of snapshots that
this ``tmin``/``tmax`` caused to be loaded will cause such an error. One can check the loaded
range of snapshots with:

.. code-block:: python

    exp_pot.tmin_exp
    exp_pot.tmax_exp

``tmin`` and ``tmax`` should not be passed for single-snapshot coefficient files.

----------
File Paths
----------

`~gala.potential.potential.EXPPotential` takes ``config_file`` and ``coef_file`` as file path
arguments. These can be absolute paths, or paths relative to the current working
directory.

The config file itself may reference file paths like the ``modelname`` and ``cachename``.
These paths can be absolute paths, or paths **relative to the config file**.

-------
Testing
-------
The tests for EXP are all in the dedicated `test_exp.py <https://github.com/adrn/gala/blob/main/tests/potential/potential/test_exp.py>`_
file. The EXP tests will be run by default if Gala was built with EXP (use ``GALA_FORCE_EXP_TEST=1`` to always test EXP).
Similarly, some of the tests will compare against pyEXP if it is available (use ``GALA_FORCE_PYEXP_TEST=1`` to always test this).

With the test dependencies installed (see :doc:`/testing`), to run just the EXP tests, one can run the following from the
repo root:

.. code-block::

    pytest tests/potential/potential/test_exp.py

--------------------
Composite Potentials
--------------------

`~gala.potential.potential.EXPPotential` fully supports composite potentials, including
mixing static and time-evolving potentials.  The potentials will be combined at the C level
as a :class:`~gala.potential.potential.CCompositePotential` when possible.
See :ref:`_compositepotential` for more info.

-----------
Limitations
-----------
The `~gala.potential.potential.EXPPotential` currently has the following limitations:

* Hessian evaluation is not supported.
* Pickling, saving, and loading is not supported.
* Performance may currently not be as high as native Gala potentials

.. TODO (adrn): any other notable limitations?

---
API
---

See :class:`~gala.potential.potential.EXPPotential` for the complete API documentation.
