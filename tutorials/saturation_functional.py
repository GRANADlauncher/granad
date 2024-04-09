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

# ## Saturation functional
#
# This example demonstrates how to preserve Pauli statistics with the saturation functional approach detailed in https://link.aps.org/doi/10.1103/PhysRevA.109.022237
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

# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Chain(16),
    lattice_type=granad.LatticeType.CHAIN,
    lattice_edge=granad.LatticeEdge.NONE,
    lattice_constant=1.42,
)
sb.add("pz", graphene)

hopping_graphene = granad.LatticeCoupling(
    orbital_id1="pz", orbital_id2="pz", lattice=graphene, couplings=[0, -2.66]
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

# create the stack object
stack = sb.get_stack( from_state = 0, to_state = 2)

# -

# We define the external illumination

# +

amplitudes = [0.0, 0, 0]
frequency = 1
peak = 2
fwhm = 0.5

# choose an x-polarized electric field propagating in z-direction
field_func = granad.electric_field_pulse(
    amplitudes, frequency, stack.positions[0, :], peak, fwhm

)
# -

# We inform GRANAD about the dissipation rates of different channels by defining a matrix gamma_matrix, where gamma_matrix[i,j] describes the transition rate from energy eigenstate i to j. In this way, we can also model gain, but we don't do this here.

# +

# propagate in time
gamma = 10
time_axis = jnp.linspace(0, 1/gamma, 200000)

# allow transfer from higher energies to lower energies only if the
# two energy levels are not degenerate
diff = stack.energies[:,None] - stack.energies
gamma_matrix = gamma * jnp.logical_and( diff < 0, jnp.abs(diff) > stack.eps, )

# -

# We now propagate the stack in time. However, in contrast to the phenomenological relaxation functional, we now choose a saturation functional that turns of decay channels if the target level is completely occupied (occupation == 2 for spin-polarized analysis). We choose a smooth approximation to the Heavside function.

# +

relaxation_function = granad.lindblad( stack, gamma_matrix.T, saturation = lambda x : 1/ ( 1 + jnp.exp( -1e6*(2.0 - x)) ) )

# run the simulation and extract occupations with a suitable postprocessing function
time_axis = jnp.linspace(0, 1/gamma, 2000000)
saveat = time_axis[::100]
stack_new, sol = granad.evolution(
    stack, time_axis, field_func, relaxation_function, saveat = saveat)

occupations = jnp.diagonal( stack.eigenvectors.conj().T @ sol.ys @ stack.eigenvectors, axis1=1, axis2=2 ).real


# plot occupations as function of time
for i, occ in enumerate(occupations.T):
    plt.plot( saveat * gamma, stack.electrons * occ.real, label = round(stack.energies[i],2) )
plt.legend()
plt.show()
plt.close()

# -
