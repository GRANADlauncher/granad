#!/bin/bash

pip install -e .
pip install mkdocs
pip install mkdocstrings
pip install mkdocstrings-python
pip install jupytext
mkdir site/tutorials
jupytext --to ipynb tutorials/*py
jupytext --to md tutorials/*ipynb; mv tutorials/*md site/tutorials; mkdocs build
