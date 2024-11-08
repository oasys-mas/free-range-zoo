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

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "free-range-zoo"
copyright = "2024, University of Georgia, University of Nebraska, Oberlin College"
author = "Ceferino Patino <cpatino2@huskers.unl.edu> Daniel Redder <daniel.redder@uga.edu>"

# The full version, including alpha/beta/rc tags
release = '1.0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_github_changelog",
]

templates_path = ["_templates"]

exclude_patterns = []

# -- Options for Napoleon -------------------------------------------------
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
napoleon_custom_sections = [("Returns", "params_style")]

# -- Options for Autodoc -------------------------------------------------
autoclass_content = "both"
autodoc_preserve_defaults = True

# -- Options for Intersphinx -----------------------------------------------
intersphinx_disabled_reftypes = ["*"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "free-range-zoo Documentation"
html_baseurl = ""
html_copy_source = False
html_favicon = "_static/img/favicon.png"
html_theme_options = {
    "light_logo": "img/darkgoat.png",
    "dark_logo": "img/lightgoat.png",
    "versioning": True,
    "description": "A FreeRangeZoo extension for Openness"
}

html_static_path = ["_static"]
html_css_files = []

# -- Options for MyST parser -------------------------------------------------
myst_heading_anchors = 3

# -- Generate Changelog -------------------------------------------------
sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
