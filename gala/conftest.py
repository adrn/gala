# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

import os

from astropy.version import version as astropy_version

from pytest_astropy_header.display import (
    PYTEST_HEADER_MODULES,
    TESTED_VERSIONS,
    pytest_report_header as astropy_header)


def pytest_configure(config):

    config.option.astropy_header = True

    # Customize the following lines to add/remove entries from the list of
    # packages for which version numbers are displayed when running the tests.
    PYTEST_HEADER_MODULES.pop('Pandas', None)
    PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'

    from . import __version__
    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = __version__


def pytest_report_header(config):
    from gala._cconfig import GSL_ENABLED

    if GSL_ENABLED:
        hdr = " +++ Gala compiled with GSL +++"
    else:
        hdr = " --- Gala compiled without GSL ---"

    return hdr + "\n"


# Uncomment the last two lines in this block to treat all DeprecationWarnings as
# exceptions. For Astropy v2.0 or later, there are 2 additional keywords,
# as follow (although default should work for most cases).
# To ignore some packages that produce deprecation warnings on import
# (in addition to 'compiler', 'scipy', 'pygments', 'ipykernel', and
# 'setuptools'), add:
#     modules_to_ignore_on_import=['module_1', 'module_2']
# To ignore some specific deprecation warning messages for Python version
# MAJOR.MINOR or later, add:
#     warnings_to_ignore_by_pyver={(MAJOR, MINOR): ['Message to ignore']}
# from astropy.tests.helper import enable_deprecations_as_exceptions  # noqa
# enable_deprecations_as_exceptions()
