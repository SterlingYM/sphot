# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'Sphot'
copyright = '2024, Y.S.Murakami'
author = 'Y.S.Murakami'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'sphinxcontrib.mermaid',
]

# Mermaid: rounded boxes + reasonable default font on the RTD theme.
mermaid_init_js = (
    "mermaid.initialize({"
    "startOnLoad: true,"
    "theme: 'default',"
    "flowchart: {curve: 'basis', htmlLabels: true, useMaxWidth: true}"
    "});"
)
# sphinxcontrib-mermaid pulls d3 in as a top-level <script> regardless of
# whether the zoom feature is enabled (mermaid_output_format defaults to
# "raw"). nbsphinx then adds RequireJS at the default priority 500. With
# the default load order, d3 lands AFTER RequireJS and tries to register
# as an anonymous AMD module — RequireJS rejects that with "Mismatched
# anonymous define()", which propagates as an unhandled error and halts
# the rest of the page-init pipeline (most mermaid blocks get stuck on
# their placeholders). Forcing d3 to load before RequireJS sidesteps the
# detection: d3 registers as a window global, then RequireJS loads and
# leaves it alone.
mermaid_js_priority = 100
templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'build',         # stray sphinx output checked into working tree
    'build_old',     # ditto
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
]
napoleon_custom_sections = [('Returns', 'params_style')]
# Most docstrings use prose types like ``data (2d array)``. Without this,
# napoleon would emit ``:type:`` fields and Sphinx would try to cross-reference
# "2d array" as a real Python class, producing dozens of nitpicky warnings.
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = False
language = 'python'

# -- Autodoc -----------------------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# autodoc_mock_imports recursively mocks submodules, so listing the top-level
# package is enough — subpackages like astropy.modeling.fitting are covered
# automatically.
autodoc_mock_imports = [
    'numpy',
    'scipy',
    'astropy',
    'petrofit',
    'h5py',
    'pandas',
    'cv2',
    'photutils',
    'tqdm',
    'csaps',
    'skimage',
    'matplotlib',
    'rich',
    'termcolor',
]

# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Path setup --------------------------------------------------------------

import os
import sys

# Route sphot's logger to a real stdlib NullHandler during doc builds, so the
# mocked rich.logging.RichHandler is never instantiated. Without this, autodoc
# import-time logging.info() calls hit MagicMock attributes inside the stdlib
# logging machinery (record.levelno >= handler.level) and TypeError out.
os.environ.setdefault('SPHOT_QUIET_IMPORT', '1')

sys.path.insert(0, os.path.abspath('../'))

# -- nbsphinx ----------------------------------------------------------------
# Notebooks are optional in API-only builds; skip them gracefully if pandoc
# is not available rather than failing the whole build.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'photutils': ('https://photutils.readthedocs.io/en/stable/', None),
}

# Silence cross-references that would resolve only if the (mocked) third-party
# package's API docs were on the path. Add new entries here if a clean
# nitpicky build is desired.
nitpick_ignore = [
    ('py:class', 'petrofit.PSFConvolvedModel2D'),
]
