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

# ## Spin orbitals
#
# This example demonstrates how to model spin orbitals in GRANAD. There are three differences to the default spin-polarized case: 1. you have to name the orbitals differently, e.g. pz_up and pz_down, 2. you leave one of it empty when adding it, 3. you pass spin_degenerate = False to get_stack.
#

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
shape=granad.Triangle(4.1),
lattice_edge=granad.LatticeEdge.ARMCHAIR,
lattice_type=granad.LatticeType.HONEYCOMB,
lattice_constant=2.46,
)
sb.add("pz_up", graphene)

# make it half full
sb.add("pz_down", graphene, occupation = 0)

uu = granad.LatticeCoupling(
    orbital_id1="pz_up", orbital_id2="pz_up", lattice=graphene, couplings=[0, -2.66]
)
dd = granad.LatticeCoupling(
    orbital_id1="pz_down", orbital_id2="pz_down", lattice=graphene, couplings=[0, -2.66]
)
ud = granad.LatticeCoupling(
    orbital_id1="pz_up", orbital_id2="pz_down", lattice=graphene, couplings=[0, -2.66]
)

sb.set_hopping(uu)
sb.set_hopping(dd)
sb.set_hopping(ud)

cuu = granad.LatticeCoupling(
    orbital_id1="pz_up",
    orbital_id2="pz_up",
    lattice=graphene,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
cdd = granad.LatticeCoupling(
    orbital_id1="pz_down",
    orbital_id2="pz_down",
    lattice=graphene,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
cud = granad.LatticeCoupling(
    orbital_id1="pz_up",
    orbital_id2="pz_down",
    lattice=graphene,
    couplings=[16.522, 8.64, 5.333],
    coupling_function=lambda d: 14.399 / d + 0j,
)
sb.set_coulomb(cuu)
sb.set_coulomb(cdd)
sb.set_coulomb(cud)

# create the stack object
stack = sb.get_stack( from_state = 0, to_state = 0,  doping = 0, spin_degenerate = False )
print(stack.rho_0.diagonal() * stack.electrons, stack.homo, stack.rho_0.diagonal().sum() * stack.electrons)
# -
