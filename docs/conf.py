# -*- coding: utf-8 -*-

import os
import pathlib
import re
import sys
import datetime
from importlib import import_module
import warnings

# Load all of the global Astropy configuration
try:
    from sphinx_astropy.conf.v1 import *  # noqa
except ImportError:
    print('ERROR: Building the documentation for Gala requires the '
          'sphinx-astropy package to be installed')
    sys.exit(1)

# Get configuration information from setup.cfg
from configparser import ConfigParser
conf = ConfigParser()

docs_root = pathlib.Path(__file__).parent.resolve()
conf.read([str(docs_root / '..' / 'setup.cfg')])
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
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
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

from cmastro import cmaps
plot_formats = [('png', 200), ('pdf', 200)]
plot_apply_rcparams = True
# NOTE: if you update these, also update docs/tutorials/nb_setup
plot_rcparams = {
    'image.cmap': 'cma:hesperia',

    # Fonts:
    'font.size': 16,
    'figure.titlesize': 'x-large',
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',

    # Axes:
    'axes.labelcolor': 'k',
    'axes.axisbelow': True,

    # Ticks
    'xtick.color': '#333333',
    'xtick.direction': 'in',
    'ytick.color': '#333333',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,

    'figure.dpi': 300,
    'savefig.dpi': 300,
}
plot_include_source = False

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
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html"]
}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = str(docs_root / '_static' / 'm104.ico')

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

# show inherited members for classes
automodsumm_inherited_members = True

# Add nbsphinx
extensions += [
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinxcontrib.bibtex',
    'rtds_action'
]

# Bibliography:
bibtex_bibfiles = ['refs.bib']
bibtex_reference_style = 'author_year'

# Custom setting for nbsphinx - timeout for executing one cell
nbsphinx_timeout = 300
nbsphinx_kernel_name = os.environ.get('NBSPHINX_KERNEL_NAME', 'python3')

# nbsphinx hacks (thanks exoplanet)
import nbsphinx
from nbsphinx import markdown2rst as original_markdown2rst

nbsphinx.RST_TEMPLATE = nbsphinx.RST_TEMPLATE.replace(
    "{%- if width %}", "{%- if 0 %}"
).replace("{%- if height %}", "{%- if 0 %}")

def subber(m):
    return m.group(0).replace("``", "`")

prog = re.compile(r":(.+):``(.+)``")

def markdown2rst(text):
    return prog.sub(subber, original_markdown2rst(text))

nbsphinx.markdown2rst = markdown2rst

# rtds-action
if "GITHUB_TOKEN" in os.environ:
    print("GitHub Token found: retrieving artifact")

    # The name of your GitHub repository
    rtds_action_github_repo = setup_cfg['github_project']

    # The path where the artifact should be extracted
    # Note: this is relative to the conf.py file!
    rtds_action_path = "tutorials"

    # The "prefix" used in the `upload-artifact` step of the action
    rtds_action_artifact_prefix = "notebooks-for-"

    # A GitHub personal access token is required, more info below
    rtds_action_github_token = os.environ["GITHUB_TOKEN"]

    # Whether or not to raise an error on ReadTheDocs if the
    # artifact containing the notebooks can't be downloaded (optional)
    rtds_action_error_if_missing = True

else:
    rtds_action_github_repo = ''
    rtds_action_github_token = ''
    rtds_action_path = ''

## -- Retrieve Zenodo record for most recent version of Gala:
zenodo_path = docs_root / 'ZENODO.rst'
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

## -- Check for executed tutorials and only add to toctree if they exist:

tutorial_files = [
    "tutorials/Milky-Way-model.ipynb",
    "tutorials/integrate-potential-example.rst",
    "tutorials/pyia-gala-orbit.ipynb",
    "tutorials/integrate-rotating-frame.rst",
    "tutorials/mock-stream-heliocentric.rst",
    "tutorials/circ-restricted-3body.rst",
    "tutorials/define-milky-way-model.ipynb",
    "tutorials/Arbitrary-density-SCF.ipynb"
]

_not_executed = []
_tutorial_toctree_items = []
for fn in tutorial_files:
    if not pathlib.Path(fn).exists() and "GITHUB_TOKEN" not in os.environ:
        _not_executed.append(fn)
    else:
        _tutorial_toctree_items.append(fn)

if _tutorial_toctree_items:
    _tutorial_toctree_items = '\n    '.join(_tutorial_toctree_items)
    _tutorial_toctree = f"""
.. toctree::
    :maxdepth: 1
    :glob:

    {_tutorial_toctree_items}
    """

else:
    _tutorial_toctree_items = 'No tutorials found!'

if _not_executed:
    print(
        "\n-------- Gala warning --------\n"
        "Some tutorial notebooks could not be found! This is likely because "
        "the tutorial notebooks have not been executed. If you are building "
        "the documentation locally, you may want to run 'make exectutorials' "
        "before running the sphinx build.")
    print(f"Missing tutorials: {', '.join(_not_executed)}\n")

with open('_tutorials.rst', 'w') as f:
    f.write(_tutorial_toctree)
