#!/bin/bash

pip install -e .
jupytext --to md tutorials/*py; mv tutorials/*md site/tutorials; mkdocs build
jupytext --to ipynb tutorials/*py
