from __future__ import absolute_import
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():
    exts = []

    # malloc
    mac_incl_path = "/usr/include/malloc"

    # it's annoying that I can't do this:
    # cfg = setup_helpers.DistutilsExtensionArgs()
    # cfg['sources'].append('gary/potential/*.pyx')
    # cfg['sources'].append('gary/potential/_cbuiltin.c')
    # cfg['include_dirs'].append('numpy')
    # cfg['include_dirs'].append(gary_incl_path)
    # cfg['include_dirs'].append(mac_incl_path)
    # cfg['extra_compile_args'].append('--std=gnu99')
    # exts.append(Extension('gary.potential.*', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gary/potential/cpotential.pyx')
    exts.append(Extension('gary.potential.cpotential', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gary/potential/cpotential.pyx')
    cfg['sources'].append('gary/potential/_cbuiltin.c')
    exts.append(Extension('gary.potential.cbuiltin', **cfg))

    return exts

def get_package_data():
    return {'gary.potential': ['*.pxd', '*.c', '*.h']}
