import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import starrygl

project = 'StarryGL'
copyright = '2023, StarryGL Team'
author = 'StarryGL Team'

version = starrygl.__version__
release = starrygl.__version__


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.viewcode",
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
