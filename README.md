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

Explore the [published documentation and tutorials](https://granadlauncher.github.io/granad/)
or their [Python sources](docs_src/tutorials).

## Contributing

See the [contributor guide](management.md) for environment setup, dependency
extras, [tests](management.md#tests), [clean-install verification](management.md#clean-installation-check),
[building documentation](management.md#documentation), and
[CI and publishing](management.md#ci-and-publishing).
