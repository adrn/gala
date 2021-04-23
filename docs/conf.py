# -*- coding: utf-8 -*-

import os
import pathlib
import sys
import datetime
from importlib import import_module
import warnings

# Get configuration information from setup.cfg
from configparser import ConfigParser
conf = ConfigParser()

conf.read([os.path.join(os.path.dirname(__file__), '..', 'setup.cfg')])
setup_cfg = dict(conf.items('metadata'))

# -- General configuration ----------------------------------------------------

# By default, highlight as Python 3.
highlight_language = 'python3'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# Don't show summaries of the members in each class along with the
# class' docstring
numpydoc_show_class_members = False

# Whether to create cross-references for the parameter types in the
# Parameters, Other Parameters, Returns and Yields sections of the docstring.
numpydoc_xref_param_type = True

# Words not to cross-reference. Most likely, these are common words used in
# parameter type descriptions that may be confused for classes of the same
# name.
numpydoc_xref_ignore = {
    'type', 'optional', 'default', 'or', 'of', 'method', 'instance', "like",
    "class", 'subclass', "keyword-only", "default", "thereof", "mixin",
    # needed in subclassing numpy  # TODO! revisit
    "Arguments", "Path",
    # TODO! not need to ignore.
    "flag", "bits",
}

# Mappings to fully qualified paths (or correct ReST references) for the
# aliases/shortcuts used when specifying the types of parameters.
# Numpy provides some defaults
# https://github.com/numpy/numpydoc/blob/b352cd7635f2ea7748722f410a31f937d92545cc/numpydoc/xref.py#L62-L94
# so we only need to define Astropy-specific x-refs
numpydoc_xref_aliases = {
    # ulta-general
    "-like": ":term:`-like`",
    # python & adjacent
    "file-like": ":term:`python:file-like object`",
    "file": ":term:`python:file object`",
    "iterator": ":term:`python:iterator`",
    "path-like": ":term:`python:path-like object`",
    "module": ":term:`python:module`",
    "buffer-like": ":term:buffer-like",
    "function": ":term:`python:function`",
    # for matplotlib
    "color": ":term:`color`",
    # for numpy
    "ints": ":class:`python:int`",
    # for astropy
    "unit-like": ":term:`unit-like`",
    "quantity-like": ":term:`quantity-like`",
    "angle-like": ":term:`angle-like`",
    "table-like": ":term:`table-like`",
    "time-like": ":term:`time-like`",
    "frame-like": ":term:`frame-like`",
    "coordinate-like": ":term:`coordinate-like`",
    "number": ":term:`number`",
    "Representation": ":class:`~astropy.coordinates.BaseRepresentation`",
    "writable": ":term:`writable file-like object`",
    "readable": ":term:`readable file-like object`",
}

autosummary_generate = True

automodapi_toctreedirnm = 'api'

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the "smart" one.
default_role = 'obj'

# Class documentation should contain *both* the class docstring and
# the __init__ docstring
autoclass_content = "both"

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog = """
"""

# intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/',
               (None, 'http://data.astropy.org/intersphinx/python3.inv')),
    'numpy': ('https://numpy.org/doc/stable/',
              (None, 'http://data.astropy.org/intersphinx/numpy.inv')),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/',
              (None, 'http://data.astropy.org/intersphinx/scipy.inv')),
    'matplotlib': ('https://matplotlib.org/',
                   (None, 'http://data.astropy.org/intersphinx/matplotlib.inv')),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
    'sympy': ('https://docs.sympy.org/latest/', None)
}

# Show / hide TODO blocks
todo_include_todos = True

# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = setup_cfg['name']
author = setup_cfg['author']
copyright = '{0}, {1}'.format(
    datetime.datetime.now().year, setup_cfg['author'])

package_name = 'gala'
import_module(package_name)
package = sys.modules[package_name]

# TODO: Use Gala style when building docs
from gala.mpl_style import mpl_style
plot_rcparams = mpl_style
plot_apply_rcparams = True
plot_formats = [('png', 200)]
plot_include_source = True

# The short X.Y version.
version = package.__version__.split('-', 1)[0]
# The full version, including alpha/beta/rc tags.
release = package.__version__


# -- Options for HTML output ---------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_logo = '_static/Gala_Logo_RGB.png'

html_theme_options = {
    "logo_link": "index",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/adrn/gala",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/adrianprw",
            "icon": "fab fa-twitter-square",
        },
    ],
}

# Add any paths that contain custom themes here, relative to this directory.
# To use a different custom theme, add the directory containing the theme.
# html_theme_path = ['_themes/sphinx_rtd_theme']

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes. To override the custom theme, set this to the
# name of a builtin theme or the name of a custom theme in html_theme_path.
# html_theme = "sphinx_rtd_theme"

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '_static'))
html_favicon = os.path.join(path, 'm104.ico')

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = '{0} v{1}'.format(project, release)

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# Static files to copy after template files
html_static_path = ['_static']
html_css_files = [
    "gala.css"
]


# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]


## -- Options for the edit_on_github extension ----------------------------------------

# show inherited members for classes
automodsumm_inherited_members = True

# Add nbsphinx
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'sphinx_astropy.ext.doctest',
    'sphinx_astropy.ext.generate_config',
    'sphinx_astropy.ext.missing_static',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'matplotlib.sphinxext.plot_directive']

# Custom setting for nbsphinx - timeout for executing one cell
nbsphinx_timeout = 300
if 'NBSPHINX_KERNEL_NAME' in os.environ:
    nbsphinx_kernel_name = os.environ['NBSPHINX_KERNEL_NAME']


## -- Retrieve Zenodo record for most recent version of Gala:
zenodo_path = pathlib.Path('ZENODO.rst')
if not zenodo_path.exists():
    import textwrap
    try:
        import requests
        headers = {'accept': 'application/x-bibtex'}
        response = requests.get('https://zenodo.org/api/records/4159870',
                                headers=headers)
        response.encoding = 'utf-8'
        zenodo_record = (".. code-block:: bibtex\n\n" +
                         textwrap.indent(response.text, " "*4))
    except Exception as e:
        warnings.warn(f"Failed to retrieve Zenodo record for Gala: {str(e)}")
        zenodo_record = ("`Retrieve the Zenodo record here "
                         "<https://zenodo.org/record/4159870>`_")

    with open(zenodo_path, 'w') as f:
        f.write(zenodo_record)
