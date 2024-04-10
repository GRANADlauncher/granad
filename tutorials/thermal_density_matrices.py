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

# ## Thermal density matrices
#
# By passing a finite value for beta, you get a thermal density matrix.
#

# +
import jax
import jax.numpy as jnp
import granad

# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Chain(10),
    lattice_type=granad.LatticeType.CHAIN,
    lattice_edge=granad.LatticeEdge.NONE,
    lattice_constant=1,
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
stack = sb.get_stack(beta=0.1)

# thermal occupation factors
granad.show_energies(stack)
# -
