.. _exp_tutorial:

==============================
Using EXP potentials with Gala
==============================

Gala supports `EXP <https://exp-docs.readthedocs.io>`_ as backend for
gravitational potentials. This requires building EXP, building Gala with EXP
support, and then setting up a `~gala.potential.EXPPotential` object using the
user's EXP config and coefficient files.

Note that EXP support currently requires building Gala (and EXP) from source.
Additionally, this workflow has only been tested on Linux, although it should be
possible to run on MacOS, too.

If something goes wrong, Gala builds and tests the EXP integration as part of CI,
so it may help to check that `recipe <https://github.com/adrn/gala/blob/main/.github/workflows/tests.yml>`_.

------------
Building EXP
------------

The `EXP documentation <https://exp-docs.readthedocs.io/en/latest/intro/install.html>`_
is the authoritative source on how to build EXP. Currently, the only Gala-specific
addition to the instructions is that Gala expects the ``build`` directory to be present
in the EXP root directory.  The ``install`` directory will be looked for in the EXP root
directory too, or one can set ``GALA_EXP_LIB_PATH`` (see below).

To install EXP's dependencies, here is one recipe that we have found to work on Ubuntu 24.04:::

    sudo apt-get install build-essential cmake gfortran git libeigen3-dev libfftw3-dev libhdf5-dev libomp-dev libopenmpi-dev ninja-build
    # install uv python, only needed if you don't already have python:
    # curl -LsSf https://astral.sh/uv/install.sh | sh

Here is another recipe using modules that has been found to work on Flatiron Institute's rusty cluster:::

    module load modules/2.3 cmake gcc openmpi hdf5 libtirpc eigen fftw git python

After installing the dependencies, one can download and build EXP with:::

    git clone --recursive https://github.com/EXP-code/EXP.git
    cd EXP
    cmake -G Ninja -B build -DCMAKE_INSTALL_RPATH=$PWD/install/lib --install-prefix $PWD/install
    cmake --build build
    cmake --install build

------------------------------
Building Gala with EXP support
------------------------------

Building Gala with the ``GALA_EXP_PREFIX`` environment variable set to the EXP root dir
will trigger compilation of the Gala's EXP Cython extensions. For example:::

    git clone https://github.com/adrn/gala.git
    cd gala
    export GALA_EXP_PREFIX=/path/to/EXP
    # only needed if install location is not $GALA_EXP_PREFIX/install
    # export GALA_EXP_LIB_PATH=/path/to/EXP-install/lib
    python -m venv .venv
    . .venv/bin/activate
    python -m pip install -ve .  # or uv pip install ...

The pip output should show a message like ``Gala: installing with EXP support``.

``GALA_EXP_LIB_PATH`` can be set if the CMake ``--install-prefix`` was set to a location
other than ``GALA_EXP_PREFIX/install``.

----------------------------------
Running Gala with an EXP potential
----------------------------------

To use an EXP potential with Gala, first you'll need a config file and coefficients
file from EXP (probably YAML and HDF5, respectively). Let's call them ``config.yml``
and ``coefs.h5``.

.. FUTURE: since the tutorials run on GH Actions, we could probably actually run EXP here

Then one set up a `~gala.potential.EXPPotential` object with:

.. code-block:: python

    import gala.potential as gp

    # TODO: fix this to use Adrian's method of setting units
    pot = gp.EXPPotential(
        # units=galactic,
        # units=exp_units,
        config_file="config.yml",
        coef_file="coefs.h5",
        stride=1,
        # TODO: time evolution
        tmin=0.02,
        tmax=10.,
        m_s=1 * u.Msun,
        r_s=1 * u.kpc,
    )

-----
Units
-----
.. TODO
