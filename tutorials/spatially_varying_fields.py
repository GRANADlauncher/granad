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

# ## Spatially varying fields
#
# This example demonstrates how to include spatially varying fields.
#
# ### Set up the Stack
#
# We look at a small chain.

# +

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import granad

# build stack
sb = granad.StackBuilder()

# add graphene
graphene = granad.Lattice(
    shape=granad.Chain(6),
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
stack = sb.get_stack(from_state=0, to_state=2)

# -

# We now define two functions: a uniform vector potential and the e field it generates.

# +


def uniform_vector_potential(r, omega):
    A_0 = jnp.ones_like(r)
    return lambda t: A_0 * jnp.sin(omega * t)


def uniform_e_field(r, omega):
    A_0 = jnp.ones_like(r)
    return lambda t: A_0.T * (omega * jnp.cos(omega * t))


frequency = 100
field_func = uniform_vector_potential(stack.positions, frequency)
e_field_func = uniform_e_field(stack.positions, frequency)

# propagate in time
gamma = 10
time_axis = jnp.linspace(0, 10 / gamma, int(1e8))
saveat = time_axis[:: int(1e3)]

# -

# For fields that can vary in space, it is important that you pass the vector potential as a function together with the arguent spatial = True.

# +

# time propagation
stack_new, sol = granad.evolution(
    stack,
    time_axis,
    e_field_func,
    granad.relaxation(gamma),
    saveat=saveat,
    spatial=False,
)

# calculate dipole moment
occupations = jnp.diagonal(sol.ys, axis1=1, axis2=2).real
dipole_moment = granad.induced_dipole_moment(stack, occupations)
plt.plot(saveat, dipole_moment, "-", label="E")

# time propagation
stack_new, sol = granad.evolution(
    stack, time_axis, field_func, granad.relaxation(gamma), saveat=saveat, spatial=True
)

# calculate dipole moment
occupations = jnp.diagonal(sol.ys, axis1=1, axis2=2).real
dipole_moment = granad.induced_dipole_moment(stack, occupations)
plt.plot(saveat, dipole_moment, "--", label="A")
plt.legend()
plt.show()

# -
