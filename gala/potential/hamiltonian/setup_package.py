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
    cfg['sources'].append('gala/potential/hamiltonian/chamiltonian.pyx')
    cfg['sources'].append('gala/potential/hamiltonian/src/chamiltonian.c')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    exts.append(Extension('gala.potential.hamiltonian.chamiltonian', **cfg))

    return exts

def get_package_data():

    return {'gala.potential.hamiltonian':
            ['src/*.h', 'src/*.c', '*.pyx', '*.pxd']}
