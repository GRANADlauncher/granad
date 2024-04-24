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

# # Plotting Tutorial
#
# We take a look at built-in plotting functions. 
#
# A typical simulation requires visualizing:

# 1. geometry
# 2. time-dependent arrays (such as dipole moments)
# 3. space-dependent arrays (such as eigenstates, charges, ...)

# All public plotting functions are associated with a list of orbitals.

### Geometry
# 
# Geometries can be visualized in 2D or 3D.

#
# +
from granad import Wave, Orbital, OrbitalList, MaterialCatalog, Rectangle

flake = MaterialCatalog.get("graphene").cut_flake( Rectangle(10, 10) )
flake.show_2d()
# -

# If we have a stack

# +
flake_shifted = MaterialCatalog.get("graphene").cut_flake( Rectangle(10, 10) )
flake_shifted.shift_by_vector( "sublattice_1", [0,0,1] )
flake_shifted.shift_by_vector( "sublattice_2", [0,0,1] )
stack = flake + flake_shifted
stack.show_3d()
# -

### Time-dependent arrays

# There is a dedicated function for processing TD simulation output

# +
help(flake.show_time_dependence)
# -

# So, lets plot the relaxing energy occupations of the first flake after exciting a HOMO-LUMO transition

# +
flake.set_excitation( flake.homo, flake.homo + 2, 1 )
time, density_matrices =  flake.get_density_matrix_time_domain(
        end_time=40,
        steps_time=1e5,
        relaxation_rate=1/10,
        illumination=Wave( [0,0,0], 0 ),
        skip=100,
)
density_matrices_e = flake.transform_to_energy_basis( density_matrices )
flake.show_time_dependence( density_matrices = density_matrices_e  )
# -

# If we want, we can check that the correct excitation is set

# +
flake.show_energies()
# -

### Space-dependent arrays

# The functions show_2d and show_3d are a bit more versatile than initially indicated. Let's see why at the 2d example

# +
help(flake.show_2d)
# -

# So, the display argument allows us to plot an arbitrary function defined on the grid spanned by the orbitals and filter it by orbital tags. Let's demonstrate this by visualizing the lowest energy one particle state of the flake

# +
flake.show_2d( display = flake.eigenvectors[:, 0] ) # the ground state is the
# -
