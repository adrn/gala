************
Installation
************

Requirements
============

This packages has the following strict requirements:

- `Python <http://www.python.org/>`_ 2.7 (untested with >3.0!)

- `Numpy <http://www.numpy.org/>`_ 1.7 or later

- `Cython <http://www.cython.org/>`_: 0.20 or later

- `Astropy <http://www.astropy.org/>`_ 0.3 or later

Installing
==========

Prerequisites
-------------

.. note:: If you are using MacOS X, you will need to the XCode command line
          tools.  One way to get them is to install `XCode
          <https://developer.apple.com/xcode/>`_. If you are using OS X 10.7
          (Lion) or later, you must also explicitly install the command line
          tools. You can do this by opening the XCode application, going to
          **Preferences**, then **Downloads**, and then under **Components**,
          click on the Install button to the right of **Command Line Tools**.
          Alternatively, on 10.7 (Lion) or later, you do not need to install
          XCode, you can download just the command line tools from
          https://developer.apple.com/downloads/index.action (requires an Apple
          developer account).

Obtaining the source packages
-----------------------------

Development repository
^^^^^^^^^^^^^^^^^^^^^^

The latest development version of stream-team can be cloned from github
using this command::

   git clone git://github.com/stream-team/stream-team.git

Building and Installing
-----------------------

To build the project (from the root of the source tree, e.g., inside
the cloned stream-team directory)::

    python setup.py build

To install the project::

    python setup.py install

Building the documentation
--------------------------

Requires a few extra pip-installable packages:

- `numpydoc`

- `sphinx-bootstrap-theme`
