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
    cfg['include_dirs'].append('gary/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gary/potential/cpotential.pyx')
    cfg['sources'].append('gary/potential/src/cpotential.c')
    exts.append(Extension('gary.potential.cpotential', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gary/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gary/potential/builtin/cybuiltin.pyx')
    cfg['sources'].append('gary/potential/builtin/src/_cbuiltin.c')
    cfg['sources'].append('gary/potential/src/cpotential.c')
    exts.append(Extension('gary.potential.builtin.cybuiltin', **cfg))

    return exts

def get_package_data():
    return {'gary.potential': ['*.h', '*.pxd', 'builtin/src/*.c',
                               'builtin/src/*.h', 'tests/*.yml']}
