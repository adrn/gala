import os

from pytest_astropy_header.display import (
    PYTEST_HEADER_MODULES,
    TESTED_VERSIONS,
)


def pytest_configure(config):

    config.option.astropy_header = True
    PYTEST_HEADER_MODULES.pop("Pandas", None)
    PYTEST_HEADER_MODULES["astropy"] = "astropy"

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
