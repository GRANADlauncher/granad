# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: base
# ---

# ## Welcome to GRANAD
#
# GRANAD (GRAphene Nanoflakes with ADatoms) is a tight-binding simulation utility geared towards the exploration of systems at the intersection of solid state physics and quantum optics.

# ### Features

# - Computes optical and electronic properties 
# - Full access to time-dependent density matrices via master equation
# - Relies on [JAX](https://jax.readthedocs.io/en/latest/) for performance and differentiability

# ### Installation
#
# If you want to just install the package, run
#
#```bash
# pip install git+https://github.com/username/repository.git
# 
# Additional goodies like Jupyter notebooks are built automatically by executing
# 
#```bash
# git https://github.com/GRANADlauncher/granad.git
# cd granad
# bash install.sh

# ### Speedrun
# 
# +
from granad import Material, Shapes
import jax.numpy as jnp

graphene = Material.get( "graphene" )

ten_angstroem_wide_triangle = 10 * Shapes.triangle

orbitals = graphene.cut_orbitals( ten_angstroem_wide_triangle, plot = True )

omegas = jnp.linspace( 0, 10, 100 )
absorption = orbitals.get_absorption_rpa( omegas )
plt.plot( omegas, absorption )
# -

