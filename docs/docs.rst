.. _gala-docs:

=================
Building the docs
=================

The documentation is built by Sphinx. To start, make sure you install all of the docs dependencies::

    pip install -e ".[docs]"

Then change directory into the ``docs/`` path. You now have to execute the
tutorials, make animations needed by the documentation, and run the docs build::

    make exectutorials
    make animations
    make html
