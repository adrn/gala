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
    cfg['sources'].append('gala/potential/potential/cpotential.pyx')
    cfg['sources'].append('gala/potential/potential/builtin/builtin_potentials.c')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    exts.append(Extension('gala.potential.potential.cpotential', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/potential/ccompositepotential.pyx')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    exts.append(Extension('gala.potential.potential.ccompositepotential', **cfg))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/potential/builtin/cybuiltin.pyx')
    cfg['sources'].append('gala/potential/potential/builtin/builtin_potentials.c')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    exts.append(Extension('gala.potential.potential.builtin.cybuiltin', **cfg))

    return exts

def get_package_data():

    return {'gala.potential.potential':
            ['*.h', '*.pyx', '*.pxd', '*/*.pyx', '*/*.pxd',
             'builtin/builtin_potentials.h',
             'builtin/builtin_potentials.c'
             'src/cpotential.h', 'src/cpotential.c',
             'tests/*.yml']}
