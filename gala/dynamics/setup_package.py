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
    cfg['include_dirs'].append('gala/integrate/cyintegrators')
    cfg['include_dirs'].append('gala/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/src/cpotential.c')
    cfg['sources'].append('gala/integrate/cyintegrators/dopri/dop853.c')
    cfg['sources'].append('gala/dynamics/lyapunov/dop853_lyapunov.pyx')
    exts.append(Extension('gala.dynamics.lyapunov.dop853_lyapunov', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['sources'].append('gala/dynamics/mockstream/_coord.pyx')
    cfg['extra_compile_args'].append('--std=gnu99')
    exts.append(Extension('gala.dynamics.mockstream._coord', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append('gala/integrate/cyintegrators')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['sources'].append('gala/potential/src/cpotential.c')
    cfg['sources'].append('gala/dynamics/mockstream/_mockstream.pyx')
    cfg['sources'].append('gala/integrate/cyintegrators/dopri/dop853.c')
    cfg['extra_compile_args'].append('--std=gnu99')
    exts.append(Extension('gala.dynamics.mockstream._mockstream', **cfg))

    return exts

def get_package_data():
    return {'gala.dynamics': ['*.pyx', '*.pxd', '*/*.pyx', '*/*.pxd']}
