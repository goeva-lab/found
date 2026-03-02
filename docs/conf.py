# :orphan:

project = "found"
copyright = "2025, goeva lab"
author = "goeva lab"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conf.py"]


html_theme = "pydata_sphinx_theme"
html_static_path = ["static"]

html_css_files = ["custom.css"]

nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}
nb_execution_timeout = -1
nb_execution_show_tb = True
nb_code_prompt_show = "show"
nb_code_prompt_hide = "hide"

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("http://scikit-learn.org/stable/", None),
    "altair": ("https://altair-viz.github.io/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/version/2.3/", None),
}

typehints_use_rtype = False
