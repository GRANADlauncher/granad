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

# ## Self-consistency computation
#
# This example demonstrates how to perform a self-consistent calculation.

# +
import granad
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
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
stack = sb.get_stack(from_state=0, to_state=2, doping=1)

# get a stack object where the hamiltonian and density matrices are self-consistently evaluated
stack_new = granad.get_self_consistent(stack)
print(jnp.linalg.norm(stack.energies - stack_new.energies))
print(jnp.linalg.norm(stack.hamiltonian - stack_new.hamiltonian))
# -
