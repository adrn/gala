from distutils.core import Extension
from collections import defaultdict


def get_extensions():
    import numpy as np

    exts = []

    # malloc
    mac_incl_path = "/usr/include/malloc"

    cfg = defaultdict(list)
    cfg['include_dirs'].append(np.get_include())
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['include_dirs'].append('gala')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/potential/cpotential.pyx')
    cfg['sources'].append('gala/potential/potential/builtin/builtin_potentials.c')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    exts.append(Extension('gala.potential.potential.cpotential', **cfg))

    cfg = defaultdict(list)
    cfg['include_dirs'].append(np.get_include())
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['include_dirs'].append('gala')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/potential/ccompositepotential.pyx')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    exts.append(Extension('gala.potential.potential.ccompositepotential', **cfg))

    cfg = defaultdict(list)
    cfg['include_dirs'].append(np.get_include())
    cfg['include_dirs'].append(mac_incl_path)
    cfg['include_dirs'].append('gala/potential')
    cfg['include_dirs'].append('gala')
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
