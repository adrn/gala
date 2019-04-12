from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():
    exts = []

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('gala')
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('gala/cconfig.pyx')
    exts.append(Extension('gala._cconfig', **cfg))

    return exts


def get_package_data():

    return {'gala': ['extra_compile_macros.h', 'cconfig.pyx']}


def get_build_options():
    return [('nogsl',
             'Install without the GNU Scientific Library (GSL)',
             True)]
