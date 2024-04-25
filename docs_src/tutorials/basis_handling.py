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

# # 10. Basis Handling
#
# We talk about how to switch between site and energy basis.

### Default basis
#
# By default, GRANAD uses the site basis. You can explicitly force a quantity to be given in energy basis by appending "_e" to it.

# +
from granad import Material

chain  = MaterialCatalog.get("metal_1d")
flake = chain.cut_flake( unit_cells = 10 )
site_occupations = flake.initial_density_matrix.diagonal() # site basis
site_occupations = flake.initial_density_matrix_e.diagonal() # energy basis
# -

# Additionally, should you be unsure, the site basis is always given by appending "_x".

# +
print(jnp.all(flake.initial_density_matrix_x == flake.initial_density_matrix))
# -

# Displaying transition dipole moments

# +
import matplotlib.pyplot as plt
plt.matshow(flake.dipole_operator_e[0].real)
plt.show()
# -





