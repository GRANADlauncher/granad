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

# ## Basis Handling
#

# ### Arrays with Basis

# GRANAD uses two bases: energy and site. You can always tell whether a given operator is in site or energy basis. To see how this works, consider the dipole operator computed from a graphene flake

from granad import Material, Shapes

graphene = Material.get( "graphene" )
ten_angstroem_wide_triangle = 10 * Shapes.triangle
flake = graphene.cut_orbitals( ten_angstroem_wide_triangle, plot = True )

flake.dipole_operator
# -

# We see that this operator is represented by a special object, ArrayWithBasis, which is just that: an array (the operator matrix representation) and a boolean variable indicating whether this particular array is in site basis. # TODO: mention that I will probably change this.

# You can use arrays with basis like regular arrays. However, the results of these operations don't have any flags anymore, i.e. this is just a JAX array element

# +
flake.dipole_operator[0,1]
# -

# This is just a regular JAX array

# +
flake.dipole_operator @ jnp.arange( len(flake) )
# -

# This is too

# +
flake.dipole_operator @ flake.dipole_operator 
# -

# ### Basis Transformations

# Often, you want to switch between bases. This happens as follows

# +
dipole_operator_in_energy_basis = flake.transform_to_energy_basis( flake.dipole_operator )
dipole_operator_in_energy_basis
# -

# If you try to convert an array that is already in a basis to that basis, nothing happens

# +
jnp.all(flake.dipole_operator == flake.transform_to_site_basis( flake.dipole_operator ) )
# -

# You don't have to create arrays with basis if you want to use the transformation function. 

# +
my_custom_site_matrix_full_of_ones = jnp.ones( len(orbitals), len(orbitals) )
my_custom_energy_matrix = flake.transform_to_energy_basis( my_custom_site_matrix_full_of_ones  )
my_custom_energy_matrix
# -

# The basis transformation also works with non-matrix arrays. Let's say you run a TD simulation and inspect its result

# +
density_matrices = flake.get_density_matrix_time_domain( start_time = 0, end_time = 10, steps_time = 100 )
density_matrices
# -

# You see this also an ArrayWithBasis, again in space basis. It is not a matrix, but has a different shape

# +
density_matrices.shape
# -

# So, the first axis is time, the second and the third are space, i.e. this here is the density matrix at the 10-th time step

# +
density_matrices[10]  # the same as density_matrices[10,:,:]
# -

# However, we can convert it like a regular operator

# +
density_matrices_in_energy_basis = flake.transform_to_energy_basis( density_matrices )
density_matrices_in_energy_basis
# -

# So, this is the density matrix at the 10-th time step in energy basis

# +
density_matrices_in_energy_basis[10]  # the same as density_matrices[10,:,:]
# -

# ### A warning

# If anything breaks, you can always access the raw array like so

# +
flake.dipole_operator.array
#-

# This is just a normal JAX array and everything should be fine. # TODO: warn mutability
