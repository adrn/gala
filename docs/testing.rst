.. _gala-test:

=================
Running the tests
=================

The tests are written assuming they will be run with `tox
<https://tox.readthedocs.io/en/latest/>`_ or `pytest <http://doc.pytest.org/>`_.

To run the tests with tox, first make sure that tox is installed;

    pip install tox

then run the basic test suite with:

    tox -e test

or run the test suite with all optional dependencies with:

    tox -e test-alldeps

You can see a list of available test environments with:

    tox -l -v

which will also explain what each of them does.

You can also run the tests directly with pytest. To do this, make sure to
install the testing requirements (from the cloned ``gala`` repository
directory)::

    pip install -e ".[test]"

Then you can run the tests with:

    pytest gala
