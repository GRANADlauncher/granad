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

# ## Dynamical multipole moments
#
# We want to track the evolution of multipole moments.
#
# ### Set up the Stack
#
# We consider a 6-atom chain like in the publication.

# +

import granad

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


sb = granad.StackBuilder()

# geometry
triangle = granad.Triangle(7.4) 
# graphene = granad.Lattice(
#     shape=triangle,
#     lattice_type=granad.LatticeType.HONEYCOMB,
#     lattice_edge=granad.LatticeEdge.ARMCHAIR,
#     lattice_constant=2.46,
# )
graphene = granad.Lattice(
    shape=granad.Chain(6),
    lattice_type=granad.LatticeType.CHAIN,
    lattice_edge=granad.LatticeEdge.NONE,
    lattice_constant=1,
)
sb.add("pz", graphene)

# couplings
hopping_graphene = granad.LatticeCoupling(
    orbital_id1="pz", orbital_id2="pz", lattice=graphene, couplings=[0, -2.66] # list of hopping amplitudes like [onsite, nn, ...]
)
sb.set_hopping(hopping_graphene)
coulomb_graphene = granad.LatticeCoupling(
    orbital_id1="pz",
    orbital_id2="pz",
    lattice=graphene,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
sb.set_coulomb(coulomb_graphene)

stack = sb.get_stack()

# -

# Now, we obtain the (matrix representations of) electric multipole moment operators

# +

dip = granad.position_operator(stack)
quad = granad.quadrupole_operator(stack)

# -

# Higher moments follow the same pattern. The results are matrices of shape NxNx3x..., where the first two components are the TB-matrix components and the last are Cartesian components indexing the particular multipole component.

# +

def vector_potential( r, omega ):
    A_0 = 1e-3 * jnp.ones_like( r )
    return lambda t : A_0 * jnp.sin( omega * t )

def uniform_e_field( r, omega ):
    A_0 = jnp.ones_like( r )
    return lambda t : A_0.T * (omega * jnp.cos( omega * t ) ) 

field_func = vector_potential( stack.positions, 1 )
field_func = uniform_e_field( stack.positions, 1 )

# run the simulation
gamma = 1
time_axis = jnp.linspace(0, 1/gamma, int(1e6))
saveat = time_axis[::100]
# stack_new, sol = granad.evolution(
#     stack, time_axis, field_func, spatial = True, saveat = saveat)
stack_new, sol = granad.evolution(
    stack, time_axis, field_func, spatial = False, saveat = saveat)

# -

# We compare the dipole components obtained in the old way (just summing density matrix components and multiplying by position) with the expectation value of the position operator. These should agree in the absence of coupled orbitals, because here x_ij = x_i \delta_ij and since Tr[r x] = r_ij x_ji = r_ii x_i

# +

occupations = jnp.diagonal(sol.ys, axis1=1, axis2=2)
dipole_moment = granad.induced_dipole_moment(stack, occupations)
dipole_moment_from_trace = jnp.einsum( 'ijk,kjr->ir', -stack.electrons*(sol.ys - granad.to_site_basis(stack, stack.rho_stat)[None,:,:]), dip )
plt.plot( saveat, dipole_moment, '-', label = 'occupations' )
plt.plot( saveat, dipole_moment_from_trace, '--', label = 'trace' )
plt.legend()
plt.show()

# -

# Now we check what happens to quadrupoles

# +

quadrupole_moments = jnp.einsum( 'ijk,kjxy->ixy', -stack.electrons*(sol.ys - granad.to_site_basis(stack, stack.rho_stat)[None,:,:]), quad )
plt.plot( saveat, dipole_moment_from_trace, '-', label = 'dip' )
plt.plot( saveat, quadrupole_moments.diagonal(axis1=1, axis2=2), '--', label = 'quad' )
plt.legend()
plt.savefig('quad.pdf')
plt.show()

# -




