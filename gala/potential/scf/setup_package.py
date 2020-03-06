# Licensed under a 3-clause BSD style license - see PYFITS.rst

import sys
from os import path
from distutils.core import Extension
from collections import defaultdict


def get_extensions():
    import numpy as np

    exts = []

    # malloc
    mac_incl_path = "/usr/include/malloc"

    # Some READTHEDOCS hacks - see
    # https://github.com/pyFFTW/pyFFTW/pull/161/files
    # https://github.com/pyFFTW/pyFFTW/pull/162/files
    include_dirs = [path.join(sys.prefix, 'include')]
    library_dirs = [path.join(sys.prefix, 'lib')]

    # all need these:
    include_dirs.extend([np.get_include(), mac_incl_path,
                         'gala', 'gala/potential'])

    cfg = defaultdict(list)
    cfg['include_dirs'].extend(include_dirs)
    cfg['library_dirs'].extend(library_dirs)
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/scf/computecoeff.pyx')
    cfg['sources'].append('gala/potential/scf/src/bfe_helper.c')
    cfg['sources'].append('gala/potential/scf/src/coeff_helper.c')
    exts.append(Extension('gala.potential.scf._computecoeff', **cfg))

    cfg = defaultdict(list)
    cfg['include_dirs'].extend(include_dirs)
    cfg['library_dirs'].extend(library_dirs)
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/potential/src/cpotential.c')
    cfg['sources'].append('gala/potential/potential/builtin/builtin_potentials.c')
    cfg['sources'].append('gala/potential/scf/bfe.pyx')
    cfg['sources'].append('gala/potential/scf/src/bfe.c')
    cfg['sources'].append('gala/potential/scf/src/bfe_helper.c')
    exts.append(Extension('gala.potential.scf._bfe', **cfg))

    cfg = defaultdict(list)
    cfg['include_dirs'].extend(include_dirs)
    cfg['library_dirs'].extend(library_dirs)
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/potential/scf/bfe_class.pyx')
    cfg['sources'].append('gala/potential/scf/src/bfe.c')
    cfg['sources'].append('gala/potential/scf/src/bfe_helper.c')
    exts.append(Extension('gala.potential.scf._bfe_class', **cfg))

    return exts


def get_package_data():
    return {'gala.potential.scf': ['*.pyx',
                                   'tests/data/*',
                                   'tests/data/*.csv',
                                   'tests/data/*.dat.gz',
                                   'tests/data/*.coeff',
                                   '*.h', '*.pyx', '*.pxd',
                                   'src/*.c', 'src/*.h']}
