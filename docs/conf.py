# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version


project = 'GRANAD'
copyright = '2023, D.Dams, A.Ghosh, K.Słowik, J.Szczuczko'
author = 'D.Dams, A.Ghosh, K.Słowik, J.Szczuczko'
version = version("granad")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
nitpicky = True


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'matplotlib.sphinxext.plot_directive',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pyramid'
html_static_path = ['_static']
hide_args = ['k_vector', 'beta']

# skip unneeded information
def process_sig(app, what, name, obj, options, signature, return_annotation):
    if signature:
        for x in hide_args:
            if x in signature:
                signature = signature.replace(', ' + x, '')
    return (signature, return_annotation)

def maybe_skip_member(app, what, name, obj, skip, options):    
    if 'dataclass' in str(obj) and name == 'replace':
        return True
    return skip

def setup(app):
    app.connect('autodoc-skip-member', maybe_skip_member)
    app.connect("autodoc-process-signature", process_sig)
