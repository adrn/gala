from __future__ import absolute_import
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():
    exts = []

    # malloc
    mac_incl_path = "/usr/include/malloc"

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gary/integrate/_leapfrog.pyx')
    exts.append(Extension('gary.integrate._leapfrog', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gary/integrate/_dop853.pyx')
    cfg['sources'].append('gary/integrate/dopri/dop853.c')
    exts.append(Extension('gary.integrate._dop853', **cfg))

    return exts

def get_package_data():
    return {'gary.integrate': ['*.pxd', '*.c', 'dopri/*']}
