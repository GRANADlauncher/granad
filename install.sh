#!/bin/bash

pip install -e .
pip install mkdocs
pip install mkdocstrings
pip install mkdocstrings-python
pip install jupytext
jupytext --to md tutorials/*py; mv tutorials/*md site/tutorials; mkdocs build
jupytext --to ipynb tutorials/*py
