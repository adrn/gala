.. _exp_tutorial:

==============================
Using EXP potentials with Gala
==============================

Gala supports `EXP <https://exp-docs.readthedocs.io>`_ as a backend for representing
flexible and time-dependent gravitational potentials, typically constructed from N-body
simulation snapshots. This requires:

#. building EXP,
#. building Gala with EXP support,
#. and setting up a `~gala.potential.EXPPotential` object using the user's EXP config and
    coefficient files.

Note that EXP support currently requires building Gala (and EXP) from source.
Additionally, this workflow has only been tested on Linux and MacOS with the setups seen
in the `GitHub actions test config file
<https://github.com/adrn/gala/blob/main/.github/workflows/tests.yml>`_.

------------
Building EXP
------------

The `EXP documentation <https://exp-docs.readthedocs.io/en/latest/intro/install.html>`_
is the authoritative source on how to build EXP. Currently, the only Gala-specific
addition to the instructions is that Gala expects the ``build`` directory to be present
in the EXP root directory.  The ``install`` directory will be looked for in the EXP root
directory too, or one can set ``GALA_EXP_LIB_PATH`` (see below).

To install EXP's dependencies, here is one recipe that we have found to work on Ubuntu 24.04::

    sudo apt-get install build-essential cmake gfortran git libeigen3-dev libfftw3-dev libhdf5-dev libomp-dev libopenmpi-dev ninja-build
    # install uv python, only needed if you don't already have python:
    # curl -LsSf https://astral.sh/uv/install.sh | sh

Here is another recipe using modules that has been found to work on Flatiron Institute's rusty cluster::

    module load modules/2.3 cmake gcc openmpi hdf5 libtirpc eigen fftw git python

EXP also builds on Mac by installing the dependencies with Homebrew::

    brew install cmake eigen fftw hdf5 open-mpi git ninja

After installing the dependencies, one can download and build EXP on Linux with::

    git clone --recursive https://github.com/EXP-code/EXP.git
    cd EXP
    cmake -G Ninja -B build -DCMAKE_INSTALL_RPATH=$PWD/install/lib --install-prefix $PWD/install
    cmake --build build
    cmake --install build

For a full example of how to build EXP on Mac, see `this build recipe
<https://gist.github.com/adrn/afd9222416e359fcef826b7988b7d69f>`_.

------------------------------
Building Gala with EXP support
------------------------------

Building Gala with the ``GALA_EXP_PREFIX`` environment variable set to the EXP root dir
will trigger compilation of the Gala's EXP Cython extensions. For example::

    git clone https://github.com/adrn/gala.git
    cd gala
    export GALA_EXP_PREFIX=/path/to/EXP

If you build and install EXP following the instructions above, the EXP libraries will be
located in EXP/install/lib and the Gala build process knows to look there by default. If
you installed EXP to a different location, you can set the ``GALA_EXP_LIB_PATH``
environment variable to point to the EXP install directory::

    # Only do this if the install location is not $GALA_EXP_PREFIX/install
    # export GALA_EXP_LIB_PATH=/path/to/EXP-install/lib

That is, ``GALA_EXP_LIB_PATH`` can be set if the CMake ``--install-prefix`` was set to a
location other than ``GALA_EXP_PREFIX/install``.

Now you can run the Gala build. For example, using uv and pip::

    uv pip install -ve .

Or with a Python virtual environment::

    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -ve .  # or uv pip install ...

In either case, the pip output should show a message like ``Gala: installing with EXP
support``.

----------------------------------
Running Gala with an EXP potential
----------------------------------

To use an EXP potential with Gala, first you'll need a config file and coefficients
file from EXP (probably YAML and HDF5, respectively). Let's call them ``config.yml``
and ``coefs.h5``. The ``config.yml`` may reference "model" and "cache" files like
``potential.model`` and ``potential.cache``; you'll need those, too.

.. FUTURE: since the tutorials run on GH Actions, we could probably actually run EXP here

Then, setting up an `~gala.potential.EXPPotential` object is as easy as specifying the
unit system and the potential object:

.. code-block:: python

    import astropy.units as u

    import gala.potential as gp
    from gala.units import SimulationUnitSystem

    exp_units = SimulationUnitSystem(mass=1.25234e11 * u.Msun, length=3.845 * u.kpc, G=1)

    exp_pot = gp.EXPPotential(
        units=exp_units,
        config_file="config.yml",
        coef_file="coefs.h5",
    )

Then one can use the potential object like any other Gala potential. For example, to integrate
and plot an orbit:

.. code-block:: python

    import gala.dynamics as gd

    w0 = gd.PhaseSpacePosition(
        pos=[-8, 0.0, 0.0] * u.kpc,
        vel=[0.0, 180, 0.0] * u.km / u.s,
    )
    orbit = gp.Hamiltonian(exp_pot).integrate_orbit(w0, dt=1 * u.Myr, t1=0, t2=1 * u.Gyr)
    fig = orbit.plot(
        ["x", "y"], units=u.kpc, linestyle="-", alpha=0.5, label="exp orbit"
    )

-----
Units
-----
.. TODO (adrn): discuss units. This could also be a paragraph in the previous section.

--------------
Time Evolution
--------------

EXP potentials are time-evolving by default. To select a single static snapshot from the
coefficients file, one can use the ``snapshot_index`` parameter:

.. code-block:: python

    pot = gp.EXPPotential(
        units=exp_units,
        config_file="config.yml",
        coef_file="coefs.h5",
        snapshot_index=0,
    )

For time-evolving potentials, if one tries to evaluate the potential outside of the time range
stored in the coefficients file (even indirectly, such as during an orbital integration), an
exception will be raised.

.. TODO(lgarrison): double check exception behavior here

If the coefficients file stores a very large time range but the user is only interested in a
smaller range, one can specify ``tmin`` and/or ``tmax`` for efficiency:

.. code-block:: python

    pot = gp.EXPPotential(
        units=exp_units,
        config_file="config.yml",
        coef_file="coefs.h5",
        tmin=1.,
        tmax=2.,
    )

Note that subsequently using a time outside this range will result in an exception.

-----------
Limitations
-----------
The `~gala.potential.EXPPotential` currently has the following limitations:

* Hessian evaluation is not supported.
* Pickling, saving, and loading is not supported.
* Performance may currently not be as high as native Gala potentials

.. TODO (adrn): any other notable limitations?
