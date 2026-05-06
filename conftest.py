import os
import sys
from pathlib import Path

from pytest_astropy_header.display import (
    PYTEST_HEADER_MODULES,
    TESTED_VERSIONS,
)

# Add test helpers to path so they can be imported
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))


def pytest_configure(config):
    config.option.astropy_header = True
    PYTEST_HEADER_MODULES.pop("Pandas", None)
    PYTEST_HEADER_MODULES["astropy"] = "astropy"

    from gala import __version__

    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = __version__


def pytest_report_header(config):
    from gala._cconfig import EXP_ENABLED, GSL_ENABLED

    hdr = []
    if GSL_ENABLED:
        hdr.append(" +++ Gala compiled with GSL +++")
    else:
        hdr.append(" --- Gala compiled without GSL ---")

    if EXP_ENABLED:
        hdr.append(" +++ Gala compiled with EXP +++")
    else:
        hdr.append(" --- Gala compiled without EXP ---")
    hdr.append("")

    return "\n".join(hdr)
