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

# import os
# import sys
# #Location of Sphinx files
# sys.path.insert(0, os.path.abspath('../'))
import sys
from unittest.mock import MagicMock
sys.modules['numpy'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['astropy'] = MagicMock()
sys.modules['petrofit'] = MagicMock()
sys.modules['h5py'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['photutils'] = MagicMock()
sys.modules['tqdm'] = MagicMock()
sys.modules['csaps'] = MagicMock()
sys.modules['skimage'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()

autodoc_mock_imports = ['numpy','scipy','astropy','petrofit','h5py','pandas','cv2','photutils','tqdm','csaps','skimage','matplotlib']
