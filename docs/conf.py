# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Sphot'
copyright = '2024, Y.S.Murakami'
author = 'Y.S.Murakami'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
    'nbsphinx',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import os
import sys
from unittest.mock import MagicMock

# import sys
# #Location of Sphinx files
sys.path.insert(0, os.path.abspath('../'))

autodoc_mock_imports = [
    'numpy','numpy.ma',
    'scipy','scipy.interpolate',
    'scipy.stats','scipy.ndimage',
    'scipy.optimize',
    'astropy','astropy.convolution',
    'astropy.nddata','astropy.units',
    'astropy.stats','astropy.io',
    'astropy.table',
    'astropy.modeling.fitting','astropy.modeling.models',
    'astropy.modeling',
    'astropy.modeling.functional_models',
    'petrofit','petrofit.modeling',
    'h5py','pandas','cv2','photutils','photutils.psf',
    'photutils.aperture','photutils.background','photutils.detection',
    'tqdm','tqdm.auto',
    'csaps','skimage',
    'matplotlib','matplotlib.pyplot','matplotlib.colors','matplotlib.cm',
    'rich','rich.progress','rich.table','rich.console',]
for mod in autodoc_mock_imports:
    sys.modules[mod] = MagicMock()


