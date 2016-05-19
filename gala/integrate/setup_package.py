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
    cfg['include_dirs'].append('gala/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/integrate/cyintegrators/leapfrog.pyx')
    cfg['sources'].append('gala/potential/src/cpotential.c')
    exts.append(Extension('gala.integrate.cyintegrators.leapfrog', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/src/cpotential.c')
    cfg['sources'].append('gala/integrate/cyintegrators/dop853.pyx')
    cfg['sources'].append('gala/integrate/cyintegrators/dopri/dop853.c')
    exts.append(Extension('gala.integrate.cyintegrators.dop853', **cfg))

    return exts

def get_package_data():
    return {'gala.integrate': ['*.pyx', '*.pxd', '*/*.pyx', '*/*.pxd',
                               'cyintegrators/*.c',
                               'cyintegrators/dopri/*.c', 'cyintegrators/dopri/*.h']}
