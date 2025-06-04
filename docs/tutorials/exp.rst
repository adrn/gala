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
addition to the instructions is that Gala expects the ``build`` and ``install``
directories to be present in the EXP root directory.

To install EXP's dependencies, here is one recipe that we have found to work on Ubuntu 24.04:::

    sudo apt-get install -y libeigen3-dev libfftw3-dev libopenmpi-dev libomp-dev libhdf5-dev

Here is another recipe using modules that has been found to work on Flatiron Institute's rusty cluster:::

    module load modules/2.3 cmake gcc openmpi hdf5 libtirpc eigen fftw git python

After installing the dependencies, one can download and build EXP with:::

    git clone --recursive https://github.com/EXP-code/EXP.git
    cd EXP
    cmake -B build --install-prefix $PWD/install
    cmake --build build
    cmake --install build

------------------------------
Building Gala with EXP support
------------------------------

Building Gala with the ``GALA_EXP_PREFIX`` environment variable set to the EXP root dir
will trigger compilation of the Gala's EXP Cython extensions. For example::

    git clone git://github.com/adrn/gala.git
    cd gala
    export GALA_EXP_PREFIX=/path/to/EXP
    python -m pip install -v .

The pip output should show a message like ``Gala: installing with EXP support``.

----------------------------------
Running Gala with an EXP potential
----------------------------------

To use an EXP potential with Gala, first you'll need a config file and coefficients
file from EXP (probably YAML and HDF5, respectively). Let's call them ``config.yml``
and ``coeffs.h5``.

.. FUTURE: since the tutorials run on GH Actions, we could probably actually run EXP here

Then one can use the following to set up a `~gala.potential.EXPPotential` object:::

    import gala.potential as gp

    pot = gp.EXPPotential(
    # units=galactic,
    # units=exp_units,
    config_file="config.yml",
    coeff_file="coeffs.h5",
    stride=1,
    # TODO: time evolution
    tmin=0.02,
    tmax=10.,
    m_s=1 * u.Msun,
    r_s=1 * u.kpc,
)
