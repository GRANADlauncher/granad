# Welcome to GRANAD

GRANAD (GRAphene Nanoflakes with ADatoms) is a tight-binding simulation utility geared towards the exploration of systems at the intersection of solid state physics and quantum optics. Here, we experiment with new stuff.

## Features

- Optical and electronic properties 
- Time-dependent density matrices via master equation
- Relies on [JAX](https://jax.readthedocs.io/en/latest/)

## Installation

Use Python 3.11 or newer in an isolated environment (venv or Conda). Clone the
repository and run:

```
python -m pip install -e .
```

This installs the runtime dependencies, including CPU JAX. For accelerator
installation, follow the [JAX instructions](https://docs.jax.dev/en/latest/installation.html).

## Getting started

After that, you can get started by exploring the tutorials.

## Lightweight numerical tests

Install the test extra and run:

```bash
python -m pip install -e '.[test]'
python -m pytest -q test/test_doc_invariants.py test/test_numerics.py
```

These small tests check the tutorials' numerical guarantees: chain spectra,
basis transformations, Rabi oscillations, relaxation, simulation outputs,
Fourier transforms, linear response, periodic bands, and XYZ round-trips.
They use analytical solutions and physical identities rather than saved output
snapshots, and require no notebook execution or documentation build. Each test
names the tutorial it covers. They are also collected by the full `pytest` suite.

The full documentation build still checks tutorial execution and presentation,
including the larger mean-field and topology examples.


## Documentation
The documentation is made with [mkdocs](https://www.mkdocs.org/). You can view it by typing

```
python -m pip install -e '.[docs]'
python -m ipykernel install --sys-prefix --name base --display-name base
bash build_docs.sh
mkdocs serve
```

For contributor setup, clean-install verification, and CI/deployment details,
see [AGENTS.md](AGENTS.md).
