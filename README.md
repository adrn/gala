Gary :man:
==========

Gary is a Python package for Galactic Astronomy and especially gravitational
dynamics.

Documentation
=============
Gary's core functionality is [documented](http://adrian.pw/gary/) in Sphinx, and tested with PyTest.

Installation
============

You'll first need to make sure you have the required packages installed (see
pip-requirements.txt). You can use `pip` to automatically install these with

    pip install -r pip-requirements.txt

Then it's just a matter of

    python setup.py install

Dependencies
============
See the [installation instructions](http://adrian.pw/gary/install.html) in the [documentation](http://adrian.pw/gary/).

Warning
=======

Note that in November/December 2015, in order to prepare for the first
formal release v0.1, a number of API-breaking improvements were implemented
that may wreak havok on your old code if you have updated since then. To
check out the pre-alpha version, clone this repository, and use the git tag
`pre-alpha`:

    git checkout tags/pre-alpha
