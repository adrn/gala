.. _gala-docs:

=================
Building the docs
=================

The documentation is built by Sphinx. To start, make sure you install all of the
docs dependencies (from the cloned ``gala`` repository directory; see
:ref:`gala-install-dev`)::

    uv sync

Then change directory into the ``docs/`` path. You now have to execute the
tutorials, make animations needed by the documentation, and run the docs build::

    uv run make exectutorials
    uv run make animations
    uv run make html
