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

# # Basis Handling
#
# We introduce the site and energy basis. 

### Default basis
#
# By default, GRANAD uses the site basis. You can explicitly force a quantity to be given in energy basis by appending "_e" to it.

# +
import jax.numpy as jnp
from granad import MaterialCatalog

chain  = MaterialCatalog.get("chain")
flake = chain.cut_flake( unit_cells = 10 )
site_occupations = flake.initial_density_matrix.diagonal() # site basis
energy_occupations = flake.initial_density_matrix_e.diagonal() # energy basis
# -

# Additionally, should you be unsure, the site basis is always given by appending "_x".

# +
print(jnp.all(flake.initial_density_matrix_x == flake.initial_density_matrix))
# -

# There is also a built-in function for basis transformation

# +
print(jnp.all(flake.transform_to_energy_basis(flake.hamiltonian) == flake.hamiltonian_e))
# -

# This is useful when transforming arrays of density matrices, because appending _e only works on attributes of the orbital list. Density matrices are a simulation output and as such not an attribute of the orbital list. Appending _e to the variable name will thus not work and a separate method is needed.


# Transition dipole moments can be displayed in energy basis

# +
import matplotlib.pyplot as plt
plt.matshow(flake.dipole_operator_e[0].real)
plt.colorbar()
plt.show()
# -
