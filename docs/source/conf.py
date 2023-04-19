# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from datetime import datetime
import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

sys.path.insert(0, os.path.abspath('../../'))
import deimos


# -- Project information -----------------------------------------------------

project = 'deimos'
version = deimos.__version__
current_year = datetime.today().year
copyright = '2019-{}, Pacific Northwest National Laboratory'.format(
    current_year)
author = 'Sean M. Colby'
rst_prolog = """
.. |current_year| replace:: {}
""".format(current_year)


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.intersphinx",
              "sphinx.ext.todo",
              "sphinx.ext.viewcode",
              "sphinx.ext.napoleon",
              "nbsphinx"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'
html_logo = 'assets/logo.png'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
    'collapse_navigation': True,
    'navigation_depth': 1
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
