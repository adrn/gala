.. _gala-test:

=================
Running the tests
=================

The tests are written assuming they will be run with `pytest
<http://doc.pytest.org/>`_ using the Astropy `custom test runner
<http://docs.astropy.org/en/stable/development/testguide.html>`_. To set up a
Conda environment to run the full set of tests, see the `environment-dev.yml
<https://github.com/adrn/gala/blob/master/environment-dev.yml>`_ environment
file. Once the dependencies are installed, you can run the tests two ways:

1. By importing ``gala``::

    import gala
    gala.test()

2. By cloning the ``gala`` repository and running::

    python setup.py test

