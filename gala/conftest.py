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
