# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

import sphinx_bootstrap_theme

sys.path.insert(0, os.path.abspath("../../"))



# -- Project information -----------------------------------------------------

project = 'Mike and James Collaboration'
copyright = '2022, James Twose'
author = 'James Twose'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx.ext.inheritance_diagram",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_gallery.load_style",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

html_theme_options = {
    "source_link_position": "footer",
    "bootswatch_theme": "simplex",
    "navbar_title": "Mike & James Collaboration",
    "navbar_sidebarrel": False,
    "bootstrap_version": "3",
    "nosidebar": True,
    "body_max_width": "100%",
    "navbar_links": [
        # ("Gallery", "examples/index"),
        # ("Tutorial", "tutorial"),
        # ("Home", "index"),
        # ("API", "api"),
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Add the 'copybutton' javascript, to hide/show the prompt in code
# examples, originally taken from scikit-learn's doc/conf.py
# def setup(app):
#     app.add_js_file("copybutton.js")
#     app.add_css_file("css/custom.css")